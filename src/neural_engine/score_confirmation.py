"""Independent readback of saved stimuli, credits, bank bytes and somatic spikes. No model imports."""
import hashlib, io, json, struct
from collections import Counter
import numpy as np


def recount(r,pins,g):
    def ok(v,s):
        if not v: raise ValueError(s)
    sha=lambda b:hashlib.sha256(b).hexdigest()
    def read(p,h):
        b=p.read_bytes(); ok(not p.is_symlink() and sha(b)==h,'pinned '+str(p)); return b
    def j(p): return json.loads(p.read_bytes())
    def ref(p): return json.loads(read(p,pins[p]))
    def unpack(b):
        return list(struct.unpack('BbB' if len(b)==3 else 'BbBBb',b))
    def load(b,key):
        with np.load(io.BytesIO(b),allow_pickle=False) as z: return z[key]
    o=r.parents[1]/'configs/confirmation/transfer'
    src=r.parents[1]/'configs/confirmation/source'
    model='202611203'
    source=ref(src/'PANELS.json')[model+'_0']
    par=ref(o/'PARITY.json')[model]; alph=source['alphabet']
    tails=load(read(src/source['wave'],pins[src/source['wave']]),'tail')
    templates={sha(t): (int(q['c']),bool(q['go'])) for t,q in zip(tails,source['borrowed'])}
    ok(len(templates)==4,'tail dictionary')
    ix=j(r/'RAW_INDEX.json'); used=set()
    def raw(f): used.add(f); return read(r/f,ix[f])
    def arr(f):
        with np.load(io.BytesIO(raw(f)),allow_pickle=False) as z:
            return {k:z[k] for k in ['sp','en','payout']}
    def truth(x,q,m):
        ok(x.shape==(288,8) and sha(x)==q['rid'],'row provenance')
        codes=(x>0)@(1<<np.arange(8))
        words=[''.join(str(a.index(int(v))) for v in codes[:141] if v in a) for a in alph]
        tail=x[141:].copy();tail[21:25]=0
        a,go=templates[sha(tail)]
        w=words[a]; rep=[v for v,n in zip(w,w[1:]) if v==n]
        ok(set(rep)=={'0','1'},'uninformative word')
        f,l=(1-2*int(t) for t in (rep[0],rep[-1]))
        ok(np.all(codes[162:166]==m) and (words,a,go,m,f,l)==
           (q['words'],q['addr'],q['go'],q['marker'],q['f'],q['r']),'physical metadata')
        four=np.flatnonzero((x>0).sum(1)==4); win=np.zeros(288,bool)
        if len(four):
            ok((np.diff(four)==1).all(),'fragmented permission signal')
            spec=par['spec']; levels=(80*x[four,spec['extra']]>x[four][:,spec['support']].max(1)).tolist()
            runs=[i for i in range(2,len(levels)) if levels[i]==levels[i-1]==levels[i-2]]
            if runs and levels[runs[0]]==(par['sign']*(1-2*a)>0):
                flip=next((i for i in runs if levels[i]!=levels[runs[0]]),None)
                if flip is not None: win[four[flip]:four[-1]+1]=True
        ok(bool(win.any())==go,'physical permission oracle')
        cue=sum(1<<k for k in par['cues'][a])
        ticket=(m<<16 | cue<<8 | (f+1)<<2 | (l+1)) if go else 0
        return f,l,go,win,ticket
    def classify(sp):
        if not np.any(sp):return None
        votes=[(int(v)>>8).bit_count()-(int(v)&255).bit_count() for v in sp if v]
        lead=next((v for v in votes if v),0)
        return (1 if lead>0 else -1) if lead*sum(votes)>0 else 0
    plans=ref(o/'PANEL_PINS.json')
    panel=j(r/'PANEL.json')
    ok((panel['core_seed'],panel['stimulus_seed'])==(202611203,202611213),'seeds')
    arms=('noise_single','noise_double','shift_single','shift_double');trainidx=(15,19,21,24)
    pp=plans[model+'_random_160'];tx=load(read(o/pp['wave'],pins[o/pp['wave']]),'x')
    xlearn=ref(r.parents[1]/'configs/confirmation/weak'/'INPUTS.json')['lessons'][0]
    words=set(xlearn['words']);pres={xlearn['prefix']};tr={}
    prior=r.parents[1]/'configs/confirmation/prior'
    prevrows=ref(prior/'PANEL.json')['rows']
    for i in (12,7):
        q=pp['rows'][i];truth(tx[i],q,160);words.update(q['words']);pres.add(sha(tx[i,:141]))
    for k,i in enumerate(trainidx,1):
        q=pp['rows'][i];oracle=truth(tx[i],q,160);tr[i]=oracle;q2=panel['train'][k-1]
        ok(q2==dict(index=i,**q) and oracle[2] and oracle[0]==(-1 if k%2 else 1)==-oracle[1] and
           not (set(q['words'])&words) and sha(tx[i,:141]) not in pres,'lesson provenance')
        words.update(q['words']);pres.add(sha(tx[i,:141]))
    oracle={};held={};pairs={};inds={}
    for m in (10,160):
        qs=panel['rows'][str(m)];xs=load(raw(f'held_{m}.npz'),'x');held[m]=xs;selected=[];seen=set()
        ok(len(xs)==len(qs)==48,'holdout cardinality')
        ok(not({w for q in qs for w in q['words']} & {w for q in prevrows[str(m)] for w in q['words']}),'historical panel words')
        for i,(q,x) in enumerate(zip(qs,xs)):
            f,l,go,win,ticket=truth(x,q,m);oracle[m,i]=(f,l,go,win,ticket)
            ok(not set(q['words'])&words and sha(x[:141]) not in pres,'holdout leakage')
            group='veto' if not go else 'agree' if f==l else 'conflict'
            key=(group,q['addr'],f)
            if group=='conflict' or key not in seen:selected.append(i)
            seen.add(key);pairs[m,i]=(sha(x[:141]),q['addr'],go)
        inds[m]=selected
    ok(inds[10]==inds[160] and len(inds[10])==24 and np.array_equal(held[10][:,:162],held[160][:,:162]) and
       np.array_equal(held[10][:,166:],held[160][:,166:]),'marker-only twins')
    expected=[]
    for k in range(5):
        if k:expected.append(['learn',k,[[a,160,trainidx[k-1]] for a in arms]])
        for a in arms:expected.append(['test',k,[[a,m,i] for m in (10,160) for i in inds[m]]])
    ok(len(expected)==24 and panel['schedule']==expected,'preregistered layout')
    checkpoints={(k,a):raw(f'bank_{k}_{a}.bin') for k in range(5) for a in arms}
    pend={(k,a):raw(f'pending_{k}_{a}.bin') for k in range(5) for a in arms if a.endswith('double')}
    ok(all(pend[0,a]==bytes(2) for a in arms if a.endswith('double')),'unexpected start candidate')
    start=prior/'bank_0_bad_gate.bin'
    ok(all(checkpoints[0,a]==read(start,pins[start])==bytes.fromhex('0aff00a001') for a in arms),'identical starting bytes')
    transitions=j(r/'TEACHING.json');ok(len(transitions)==16,'lesson count');events={(e['k'],e['arm']):e for e in transitions}
    trajectory={a:[unpack(checkpoints[k,a]) for k in range(5)] for a in arms}
    logs=j(r/'RELOAD.json');ok(len(logs)==24,'raw batches')
    scores={};answer={};outliers=0;test_veto=0;veto_spikes=0
    injections=[]; counted=Counter(); bits=Counter(); missing=Counter(); reference=None
    for bi,(entry,(phase,k,lanes)) in enumerate(zip(logs,expected)):
        ok(entry['raw']==f'raw_{bi:02}.npz' and entry['k']==k and entry['stage']==phase and
           len(entry['lanes'])==len(lanes),'trace identity')
        z=arr(entry['raw']);n=len(lanes);teach=phase=='learn'
        ok(z['sp'].shape==z['en'].shape==(48,288) and z['payout'].shape==(48,2) and z['sp'].dtype==np.uint16,'trace type')
        ok(not z['sp'][n:].any() and not z['en'][n:].any() and not z['payout'][n:].any(),'active pad')
        if teach and k==1:
            ok(all(np.array_equal(z[key][0],z[key][lane]) for key in z for lane in range(n)),'action not physically matched')
        if not teach and k==0:
            arrays=tuple(z[key].tobytes() for key in ('sp','en','payout'))
            if reference is None: reference=arrays
            ok(arrays==reference,'baseline readouts unequal')
        for jx,((a,m,i),entrylane) in enumerate(zip(lanes,entry['lanes'])):
            q=(pp['rows'] if teach else panel['rows'][str(m)])[i]
            ok(entrylane==dict(arm=a,marker=m,index=i,bank=f'bank_{k-int(teach)}_{a}.bin',rid=q['rid']),'incorrect lane')
            f,l,go,win,pay=(tr[i] if teach else oracle[m,i])
            ok(np.array_equal(z['en'][jx],win) and tuple(z['payout'][jx])==(pay,0),'invalid instrument')
            outside=bool(np.any(z['sp'][jx][~win]));outliers+=outside
            if teach:
                world,policy=a.split('_')
                targettruth=l if world=='shift' else f
                teacher=l if k==1 or world=='shift' else f
                pred=classify(z['sp'][jx][win]);act=pred if pred in (-1,1) else None
                factual=None if act is None else act==targettruth
                corrupt=world=='noise' and k==1
                if corrupt:ok(act==targettruth,'incorrect starting action');injections.append((a,k,q['rid'],act))
                supplied=not factual if corrupt else factual
                verdict=None if supplied is None else ((act if supplied else -act)==teacher)
                old,new=(checkpoints[t,a] for t in (k-1,k))
                owner,x,p=struct.unpack('=BbB',old[:3])
                child,y=struct.unpack('=Bb',old[3:]) if len(old)==5 else (None,None)
                ok(owner==10 and p==0 and (child is None or child==160),'unexpected bank form')
                delta=teacher*(1 if f>l else -1);value=x if child is None else y
                priorp=pend[k-1,a] if policy=='double' else b''
                nextp=bytes(2) if policy=='double' else b''
                reason='ordinary';passed=verdict is not False
                if not passed:reason='reject'
                elif policy=='double' and value*delta<0:
                    candidate=bytes((160,delta%256))
                    if verdict is None:passed=False;reason='no-bit-for-weakening'
                    elif priorp!=candidate:passed=False;nextp=candidate;reason='defer'
                    else:reason='confirmed'
                if not passed:want=old
                elif child is not None:
                    val=y+delta
                    want=old[:3] if x*y>0 and x*val>0 and abs(val)>abs(y) and abs(val)>=2 else old[:4]+bytes([val%256])
                else:
                    ok(x*delta>=0,'unexpected absent-child conflict')
                    want=struct.pack('=BbB',owner,x+delta,0)
                ok(new==want and (policy!='double' or pend[k,a]==nextp),'contact/pending oracle')
                e=events.pop((k,a))
                ok(e==dict(k=k,arm=a,rid=q['rid'],ticket=pay,teacher=teacher,action=act,physical_target=targettruth,
                    factual=factual,delivered=supplied,injected=corrupt,verdict=verdict,reason=reason,pending0=priorp.hex(),
                    pending1=nextp.hex(),before=old.hex(),after=want.hex()),'action/bit/transaction oracle')
                counted[a]+=int(passed);bits[a]+=int(supplied is not None);missing[a]+=int(act is None)
                continue
            group='veto' if not go else 'agree' if f==l else 'conflict'
            pred=classify(z['sp'][jx]);test_veto+=not go;veto_spikes+=int(not go and pred is not None)
            c=scores.setdefault(f'{k}/{a}/{m}/{group}',Counter())
            c['n']+=1;c['first']+=int(pred==f and not outside);c['last']+=int(pred==l and not outside)
            c['silent']+=int(pred is None);c['ambiguous']+=int(pred==0);c['outside']+=int(outside)
            if group=='conflict':
                answer.setdefault((k,a,m),{})[pairs[m,i]]=(pred==l and not outside,pred==f and not outside)
    ok(not events and used==set(ix),'unmatched data')
    ok(len(injections)==2 and {a for a,k,rid,act in injections}=={'noise_single','noise_double'} and
       len({(k,rid,act) for a,k,rid,act in injections})==1,'not exactly one matched corruption')
    k1rows={e['arm']:e for e in transitions if e['k']==1}
    ok(len({(e['ticket'],e['action'],e['teacher'],e['delivered']) for e in k1rows.values()})==1 and
       all(e['delivered'] is False and e['action']==-e['teacher'] and e['verdict'] is True for e in k1rows.values()),'world leak or unequal K1')
    joint={};axis={};retention={}
    for a in arms:
        for k in range(5):
            xx=answer[k,a,10];yy=answer[k,a,160]
            ok(set(xx)==set(yy) and len(xx)==16,'unpaired conflict histories')
            joint[f'{k}/{a}']=sum(xx[h][0] and yy[h][0 if k and a.startswith('shift') else 1] for h in xx)
            for m,name,metric in ((10,'X','last'),(160,'Y','first')):
                if m==160 and k and a.startswith('shift'):metric='last'
                prefix=f'{k}/{a}/{m}/'
                c,agree,veto=(scores[prefix+t] for t in ('conflict','agree','veto'))
                ok((c['n'],agree['n'],veto['n'])==(16,4,4),'denominator')
                axis[a,k,name]=bool(c[metric]==16 and agree[metric]==4 and veto['silent']==4 and
                                     not any(scores[prefix+t]['outside'] for t in ('conflict','agree','veto')))
        retention[a]={}
        for name in ('X','Y'):
            flags=[axis[a,k,name] for k in range(5)]
            stable=next((k for k in range(1,5) if all(flags[k:])),None)
            retention[a][name]=dict(pass_by_K=flags,baseline=flags[0],all_correct=all(flags),sustained_from_lesson=stable)
    return dict(first_lesson=k1rows,trajectory=trajectory,joint_true=joint,retention=retention,score=scores,
        accepted=dict(counted),delivered_bit_counts=dict(bits),no_action_counts=dict(missing),injections=injections,
        outside_emission_lanes=outliers,veto=test_veto,veto_emissions=veto_spikes,energy_J=None)
