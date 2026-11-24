"""Read-only reconstruction of stimulus matrices, spike direction and decision. No engine, no numpy."""
import hashlib,itertools,json,struct,sys,math
from pathlib import Path

from .paths import ROOT
R=ROOT/'results/flash'
def need(v,m):
    if not v: raise ValueError(m)
def file_sum(n): return hashlib.sha256((R/n).read_bytes()).hexdigest()
def read(n):return json.loads((R/n).read_bytes())

def run():
    panels=read('PANELS.json');seeds=list(range(22541,22549));kinds=['motion','arrival','last_motion','last_arrival']
    grand={a:{k:{} for k in kinds} for a in ['blind','x','xyz']};wins=0;trace_diffs=0;ballot_diffs=0
    for s in seeds:
        p=panels[str(s)];rows=p['rows']; order=sorted(range(16),key=lambda c:p['positions'][c][0])
        fp=hashlib.sha256(); twins={}
        need(len(rows)==96,'number of rows')
        for i,row in enumerate(rows):
            y=row['y'];pix=row['pixel'];k=row['kind'];t=row['onset'];w=row['dwell'];amp=row['amp']
            need(y in (-1,1) and all(b-a==y for a,b in zip(pix,pix[1:])) and len(pix)==4 and all(0<=n<8 for n in pix),'direction of physical path')
            need(4<=t<=8 and 9<=w<=13 and 3.2<=amp<4.8,'sample domain')
            buf=bytearray(80*8*8)
            for j,b in enumerate(pix):
                if not k.startswith('last_') or j==3:
                    for u in range(t+j*w,t+(j+1)*w):struct.pack_into('<d',buf,8*(8*u+b),amp)
            need(hashlib.sha256(buf).hexdigest()==p['sample_sha256'][i],'stimulus')
            fp.update(buf);twins.setdefault((k,row['pair']),{})[y]=i
        need(fp.hexdigest()==p['input_sha256'],'concatenated tensor')
        need(len(twins)==48 and all(set(q)=={-1,1} for q in twins.values()),'balanced twins')
        for (k,j),q in twins.items():
            rl,rr=[rows[q[y]] for y in (-1,1)]
            need(all(rl[key]==rr[key] for key in ('amp','onset','dwell')),'unequal pair dosing')
            if k.endswith('motion'):need(rl['pixel']==rr['pixel'][::-1],'unequal traversal')
            else:need(rl['pixel'][-1]==rr['pixel'][-1] and rl['pixel'][-1] in (3,4),'endpoint')
        lower={a:sum(1<<b for b in order[:i]) for i,a in enumerate(order)}
        upper={a:65535^((1<<a)|lower[a]) for a in order}
        def direction(mask):
            # Iterate later set bits; count earlier cells to their left/right using integer masks.
            v=0
            for earlier,later in zip(mask,mask[6:]):
                while later and earlier:
                    bit=later&-later;i=bit.bit_length()-1;later ^=bit
                    v+=(earlier&lower[i]).bit_count()-(earlier&upper[i]).bit_count()
            return v
        raw=read('seed_'+str(s)+'.json'); ab=raw['arms'];dec={}
        for a in grand:
            masks=ab[a]['spike_mask_by_sample_tick'];need(len(masks)==len(rows) and all(len(m)==80 and all(0<=v<=65535 for v in m) for m in masks),'mask domain')
            ev=[direction(m) for m in masks];dec[a]=[(v>0)-(v<0) for v in ev]
            need(ev==ab[a]['score'],'all signed scores')
            for k in kinds:
                ii=[i for i,r in enumerate(rows) if r['kind']==k]
                correct={i:rows[i]['y']*ev[i]>0 for i in ii}
                d=dict(correct=sum(correct.values()),right=sum(correct[i] for i in ii if rows[i]['y']==1),
                    left=sum(correct[i] for i in ii if rows[i]['y']==-1),ties=sum(ev[i]==0 for i in ii),
                    silent=sum(not any(masks[i]) for i in ii),spikes=sum(v.bit_count() for i in ii for v in masks[i]),n=len(ii),
                    pairs_both=sum(all(correct[x] for x in twins[k,j].values()) for j in range(12)))
                need(d==ab[a]['metrics'][k],'seed metrics')
                for key,v in d.items():grand[a][k][key]=grand[a][k].get(key,0)+v
            for j in range(12):
                q=twins['last_arrival',j];need(masks[q[-1]]==masks[q[1]],'identical endpoint null failed')
            need(ab[a]['work']['plastic_slots']==ab[a]['work']['teacher_dose']==0,'frozen')
        ix=[i for i,r in enumerate(rows) if r['kind']=='motion']
        dif=sum((rows[i]['y']==dec['xyz'][i])-(rows[i]['y']==dec['blind'][i]) for i in ix)
        td=sum(ab['xyz']['spike_mask_by_sample_tick'][i]!=ab['blind']['spike_mask_by_sample_tick'][i] for i in ix)
        bd=sum(dec['xyz'][i]!=dec['blind'][i] for i in ix)
        need(raw['contrast']=={'delta_correct':dif,'different_motion_traces':td,'different_motion_decisions':bd},'paired contrast')
        wins+=dif>0;trace_diffs+=td;ballot_diffs+=bd
    summ=read('RESULT.json')['summary'];need(grand==summ['aggregate'] and wins==summ['wins'],'aggregate')
    m=grand['xyz']['motion'];delta=100*(m['correct']-grand['blind']['motion']['correct'])/m['n']
    screen=(m['correct']/m['n']>=.75 and delta>=5 and wins>=6 and all(grand['xyz'][k][di]>=64 for k in ('motion','arrival') for di in ('left','right')) and
            (m['correct']-grand['xyz']['last_motion']['correct'])/m['n']>=.10 and all(grand[a]['last_arrival']['correct']<=96 for a in grand))
    need(math.isclose(delta,summ['delta_pp']) and screen==summ['screen_for_replication'],'screen')
    print(json.dumps(dict(status='SAVED_CHECK_OK',seeds=len(seeds),runs=8*3,rows=8*3*96,
        correct={a:{k:m['correct'] for k,m in v.items()} for a,v in grand.items()},xyz_wins=wins,
        different_motion_traces=trace_diffs,different_motion_decisions=ballot_diffs,screen=screen,engine_runs=0),sort_keys=True))

if __name__=='__main__':run()
