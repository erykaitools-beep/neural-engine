"""Reference driver: write only to a newly created destination chosen by the CLI."""
from pathlib import Path
import io, json, numpy as np
from contextlib import contextmanager
from . import credit_port, writer, stimuli, layout, merge, outcome_port, confirm, circuit
from .association import Durable
from .integrity import need, sealed_bytes, digest
from .paths import ROOT

@contextmanager
def counted_steps(g):
    original=circuit.Network.step
    def count(net,*args,**kwargs):
        result=original(net,*args,**kwargs)
        g.steps+=1; g.cell_updates+=net.s.size; g.check()
        return result
    circuit.Network.step=count
    try: yield
    finally: circuit.Network.step=original

def run(g,R):
    def put(name,b):
        with (R/name).open('xb') as f: f.write(b)
        return digest(b)
    def dump(name,obj): return put(name,json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
    rd=lambda p:json.loads(p.read_bytes())
    get=lambda p:p.read_bytes()
    cfg=ROOT/'configs/confirmation'; sk='202611203'
    pair=get(cfg/'pair.bin'); state=rd(ROOT/'configs/rhythm/start.json')[sk]
    live=state['cp'],state['contexts']['0'],rd(cfg/'gate.json')
    branches=layout.ARMS;files={};log=[];lessons=[];known={}
    initial={a:get(cfg/'prior/bank_0_bad_gate.bin') for a in branches}
    need(set(initial.values())=={bytes.fromhex('0aff00a001')},'K0 starting twins')
    desc={a:dict(file='bank_0_bad_gate.bin',sha=digest(v)) for a,v in initial.items()}
    src=rd(cfg/'source/PANELS.json')[sk+'_0']
    corpus=rd(cfg/'transfer/PANEL_PINS.json')[sk+'_random_160']
    with np.load(cfg/'transfer'/corpus['wave'],allow_pickle=False) as z:tx=z['x']
    tq=corpus['rows'];hist=rd(cfg/'weak/INPUTS.json')['lessons'][0]
    used_words=set(hist['words']);used_pref={hist['prefix']}
    for k,i in enumerate((12,7)+layout.ROWS):
        x,q=tx[i],tq[i];f=(-1,1)[k%2]
        need(q['go'] and (q['f'],q['r'],q['marker'])==(f,-f,160) and digest(x)==q['rid'] and
             not (set(q['words'])&used_words) and digest(x[:141]) not in used_pref,'training panel')
        used_words.update(q['words']);used_pref.add(digest(x[:141]))
    def record(name,z):
        buf=io.BytesIO();np.savez_compressed(buf,**z);files[name]=put(name,buf.getvalue())
    held={};priorrows=rd(cfg/'prior/PANEL.json')['rows'];samepast={}
    for m in (10,160):
        x,qs=stimuli.build(202611213,'random',m,src,get(cfg/'source'/src['wave']))
        need(len(x)==len(qs)==48 and all(digest(v)==q['rid'] and digest(v[:141]) not in used_pref and
            not set(q['words'])&used_words for v,q in zip(x,qs)),'holdout contamination')
        samepast[str(m)]=len({q['rid'] for q in qs}&{q['rid'] for q in priorrows[str(m)]})
        need(not samepast[str(m)] and not ({w for q in qs for w in q['words']} & {w for q in priorrows[str(m)] for w in q['words']}),'previous held waves or words')
        held[m]=(x,qs);record(f'held_{m}.npz',dict(x=x))
    batches=layout.batches(held[10][1])
    need(layout.selected(held[10][1])==layout.selected(held[160][1]) and len(batches)==24,'schedule')
    dump('PANEL.json',dict(core_seed=int(sk),stimulus_seed=202611213,rows={str(m):v[1] for m,v in held.items()},
         train=[dict(index=i,**tq[i]) for i in layout.ROWS],schedule=batches,old_wave_overlap=samepast))
    dump('BOUNDARIES.json',dict(latch=credit_port.controls(),bit_port=outcome_port.fixtures(),policy='both_defer',confirmation_controls=confirm.fixtures(),merge_controls=merge.controls(),imported_checkpoint=desc))
    def bind(k,a,b):
        name=f'bank_{k}_{a}.bin';h=put(name,b);files[name]=h
        known[(k,a)]=h
    def restore(k,a):
        name=f'bank_{k}_{a}.bin'
        return sealed_bytes(R/name,known[k,a])
    def persist(k,a,p):
        name=f'pending_{k}_{a}.bin';files[name]=put(name,p)
    for a in branches:
        bind(0,a,initial[a])
        if a.endswith('_double'):persist(0,a,confirm.EMPTY)
    fwd=credit_port.build(pair)
    with counted_steps(g):
        for bi,(stage,k,lanes) in enumerate(batches):
            teaching=stage=='learn';x=np.zeros((48,288,8));before=[];views=[];rows=[]
            for j,(a,m,i) in enumerate(lanes):
                ck=k-int(teaching);b=restore(ck,a);before.append(b)
                if a.endswith('_double'):
                    name=f'pending_{ck}_{a}.bin';sealed_bytes(R/name,files[name])
                xx,q=(tx[i],tq[i]) if teaching else (held[m][0][i],held[m][1][i])
                x[j]=xx;need(digest(x[j])==q['rid'],'wave binding')
                rows.append(dict(arm=a,marker=m,index=i,rid=q['rid'],bank=f'bank_{ck}_{a}.bin'))
                views.append(writer.Bank(b,diverse=True))
            views.extend(writer.Bank(bytes(3),diverse=True) for _ in range(48-len(views)))
            snapshot=[v.blob for v in views];x.flags.writeable=False
            need(bi<24 and g.steps==bi*288 and g.cell_updates==bi*288*48*16,'budget')
            out,port=fwd(Durable(pair),live,x,views)
            need(g.steps==(bi+1)*288 and len(port.rows)==48 and
                all(row.bank is v and row.bank.blob==b for row,v,b in zip(port.rows,views,snapshot)),'frozen bank / calls')
            fn=f'raw_{bi:02}.npz';record(fn,out);log.append(dict(stage=stage,k=k,lanes=rows,raw=fn))
            if teaching:
                inputs={}
                for j,(a,m,i) in enumerate(lanes):
                    q=tq[i]
                    teacher=int(q['r'] if k==1 or a.startswith('shift_') else q['f'])  # shared bad suggestion at K1; f!=r for all four lessons
                    ticket,dup=map(int,out['payout'][j]);need(ticket and not dup,'missing permission')
                    sp=out['sp'][j];en=out['en'][j].astype(bool);need(not sp[~en].any(),'premature action')
                    votes=[(int(v)>>8).bit_count()-(int(v)&255).bit_count() for v in sp[en] if v]
                    first=next((v for v in votes if v),0)
                    action=(1 if first>0 else -1) if first*sum(votes)>0 else None
                    # Environment AFTER action. Never called in test, never supplies correct sign to gate.
                    # Commit observed action for every arm before supplying any training feedback.
                    inputs[a]=(ticket,teacher,action)
                if k==1:
                    need(len(set(inputs.values()))==1 and action in (-1,1) and action==int(tq[i]['f']) and teacher==-action,'matched correct action and incorrect teacher required')
                for j,(a,m,i) in enumerate(lanes):
                    world,policy=a.split('_');ticket,teacher,action=inputs[a]
                    truth=int(tq[i]['r'] if world=='shift' else tq[i]['f'])
                    factual=None if action is None else bool(action==truth)
                    injected=(world=='noise' and k==1)
                    success=(not factual) if injected else factual
                    eid=(a,k,tq[i]['rid'])
                    verdict=outcome_port.PostAction(eid,action).consume(eid,teacher,success)
                    if policy=='double':
                        name=f'pending_{k-1}_{a}.bin';pending=sealed_bytes(R/name,files[name])
                        b,p,reason=confirm.apply(before[j],pending,ticket,teacher,verdict)
                        persist(k,a,p)
                    else:
                        b=merge.deliver(before[j],ticket,teacher if verdict is not False else None)
                        pending=p=b'';reason='reject' if verdict is False else 'ordinary'
                    bind(k,a,b)
                    lessons.append(dict(k=k,arm=a,rid=tq[i]['rid'],ticket=ticket,teacher=teacher,action=action,physical_target=truth,
                        factual=factual,delivered=success,injected=injected,verdict=verdict,reason=reason,pending0=pending.hex(),pending1=p.hex(),
                        before=before[j].hex(),after=b.hex()))
            g.check(True)
    need(g.cell_updates==24*48*288*16 and len(lessons)==16,'coverage')
    dump('TEACHING.json',lessons);dump('RELOAD.json',log);dump('RAW_INDEX.json',files)
    return dict(batches=24,reference_steps=g.steps,cell_steps=g.cell_updates,lessons=4,lesson_lanes=16,test_lanes=960,pad=176,core_training=False,energy_J=None)
