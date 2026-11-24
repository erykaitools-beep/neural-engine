"""N1.9C sequence protocol. Runs start at independently archived post-B checkpoints."""
import gzip, hashlib, json, math
import numpy as np
from . import circuit as N
from .paths import ROOT
from .sparse import digest
from .spatial_protocol import stimulus, tensor_sha
SEEDS=list(range(11100,11130))
POLICIES=['immediate','delayed','time_shuffle','label_shuffle','frozen']
ARMS=['baseline','local'];POINTS=[0,24,48,96]
CRITERIA=dict(accuracy=.75,gain=.15,count=24,contrast=.10,sign_p=.05,retention_drop=.05,lost_pairs=1)
ORDERS=[(0,1,2),(1,2,0),(2,0,1),(0,2,1),(2,1,0),(1,0,2)]
SLOTS=[4,24,44,64];LABELS=[0,0,0,1,1,1];FAST=['p','d','s','bp','ca','a','inhibition']
def load(p):
    if p.name.startswith('result_'):
        with gzip.open(p.with_suffix('.json.gz'),'rt') as f: return json.load(f)
    return json.loads(p.read_bytes())
def panel(seed):return load(ROOT/'configs/cohort'/f'data_{seed}.json')
def checkpoint(seed,arm):return load(ROOT/'configs/cohort'/f'start_{seed}.json')[arm]
def sign_test(deltas):
    pos = sum(d>1e-12 for d in deltas)
    neg = sum(d< -1e-12 for d in deltas)
    n = pos+neg
    p = min(1.,2*sum(math.comb(n,k) for k in range(min(pos,neg)+1))/2**n) if n else 1.
    return dict(positive=pos,negative=neg,ties=len(deltas)-n,p_two_sided=p)
def pairs():
    return [(g*6+next(j for j in range(3) if ORDERS[j][-1]==k),g*6+next(j for j in range(3,6) if ORDERS[j][-1]==k)) for g in range(40) for k in range(3)]
def tensors(bundles):
    iso=np.stack([stimulus(e) for bb in bundles for e in bb]);n=len(bundles)
    if np.any(iso[:,:12]) or np.any(iso[:,29:]) or not np.all(np.count_nonzero(iso,axis=(1,2))==48):raise ValueError('pulse envelope')
    x=np.zeros((24*n,161,8))
    for b in range(n):
        for q in range(4):
            others=[i for i in range(4) if i!=q]
            for j,order in enumerate(ORDERS):
                for t,c in zip(SLOTS,[others[i] for i in order]+[q]):x[b*24+q*6+j,t:t+17]=iso[b*4+c,12:29]
    return iso,x,np.tile(LABELS,n*4).astype(int)

def data_check(d):
    if d!=panel(d['seed']):raise ValueError('data identity')
    tr,tx,y=tensors(d['train']);te,xx,z=tensors(d['test']);old,_,_=tensors(d['old'])
    sets=[{tensor_sha(a) for a in v} for v in (tr,te,old)]
    if any(sets[a]&sets[b] for a,b in ((0,1),(0,2),(1,2))):raise ValueError('input split overlap')
    for q in range(4):
        for last in range(3):
            tab=np.zeros((2,2),int)
            for i in range(96):
                if (i//6)%4==q and ORDERS[i%6][-1]==last:tab[y[i],d['teacher_labels'][i]]+=1
            if not np.all(tab==2):raise ValueError('label null imbalance')
    for y0 in (0,1):
        if sorted(np.array(d['time_starts'])[y==y0].tolist())!=sorted([4,24,44,64,84,104,124,144]*6):raise ValueError('time null imbalance')
    if sorted(d['order'])!=list(range(96)):raise ValueError('training order')
    for xs in (tx,xx):
        for b in range(len(xs)//24):
            ref=[math.fsum(xs[b*24,:,c]) for c in range(8)]
            if any([math.fsum(a[:,c]) for c in range(8)]!=ref for a in xs[b*24:(b+1)*24]):raise ValueError('stimulation imbalance')
    return tr,tx,y,te,xx,z,old

def teach(net,x,label,start,frozen=False):
    before=net.payload();net.reset();before_w=net.w.copy();before_t=net.theta.copy()
    if frozen:
        for k in ('w','theta','neighbors'):getattr(net,k).flags.writeable=False
    for xx in x:net.step(xx,None,False)
    preview=net.count[0].copy()
    if net.payload()!=before:raise ValueError('preview mutation')
    clipped=0
    if not frozen:net.theta,clipped=N.E.feedback(net.theta,preview,'local')
    net.reset();teacher=np.zeros((1,16));teacher[:,label*8:(label+1)*8]=1.8
    peak=0.;ltp=ltd=0
    for t,xx in enumerate(x):
        used=net.step(xx,teacher if start<=t<start+16 else None,not frozen)
        peak=max(peak,float(net.ca.max()));ltp+=int(np.count_nonzero((used>=net.c['theta_p'])&(net.p[0]>1e-8)));ltd+=int(np.count_nonzero((used>=net.c['theta_d'])&(used<net.c['theta_p'])&(net.p[0]>1e-8)))
    if abs(net.teacher_dose-115.2)>1e-8:raise ValueError('teacher dose')
    if frozen and net.payload()!=before:raise ValueError('frozen mutation')
    if not all(np.isfinite(getattr(net,k)).all() for k in FAST+['w','theta']):raise ValueError('nonfinite induction')
    return dict(before=digest(before),after=digest(net.payload()),start=start,label=label,teacher_dose=net.teacher_dose,teacher_spikes=net.teacher_spikes,
      preview_spikes=int(preview.sum()),ca_peak=peak,ltp_slots=ltp,ltd_slots=ltd,weight_l1=float(np.abs(net.w-before_w).sum()),theta_l1=float(np.abs(net.theta-before_t).sum()),dw_slots=net.dw_slots,clipped=clipped)

def measures(counts,labels):
    c=np.asarray(counts);labels=np.asarray(labels);scores=c.reshape(-1,2,8).sum(-1);tie=scores[:,0]==scores[:,1];pred=scores.argmax(-1)
    return dict(correct=int((pred==labels).sum()),neutral_correct=float(((pred==labels)&~tie).sum()+.5*tie.sum()),total=len(labels),silent=int((scores.sum(-1)==0).sum()),ties=int(tie.sum()),spikes=int(c.sum()))

def validate_readout(r,p,size,old=False,enriched=False):
    expected={'counts','Q','raster_sha256','payload_sha256'}|({'metrics','tasks'} if old and enriched else {'metrics','by_Q','by_last','same_last'} if enriched else set())
    if set(r)!=expected or r['payload_sha256']!=digest(p):raise ValueError('readout schema/binding')
    c=np.asarray(r['counts'])
    if c.shape!=(size,16) or not np.issubdtype(c.dtype,np.integer) or (c<0).any() or (c>(80 if old else 97)).any():raise ValueError('count domain')
    if old:
        if r['Q'] is not None:raise ValueError('old Q field')
    else:
        q=np.asarray(r['Q'])
        if q.shape!=c.shape or not np.issubdtype(q.dtype,np.integer) or (q<0).any() or (q>17).any() or (q>c).any():raise ValueError('Q count domain')
    if len(r['raster_sha256'])!=size or any(not isinstance(h,str) or len(h)!=64 or set(h)-set('0123456789abcdef') for h in r['raster_sha256']):raise ValueError('raster hashes')

def strata(c,y):
    c,y=np.asarray(c),np.asarray(y);ix=np.arange(len(y));scores=c.reshape(-1,2,8).sum(-1);pred=scores.argmax(-1);tie=scores[:,0]==scores[:,1]
    pairs=[(g*6+a,g*6+b) for g in range(len(y)//6) for a,b in ((1,4),(2,3),(0,5))]
    return dict(by_Q={str(q):measures(c[(ix//6)%4==q],y[(ix//6)%4==q]) for q in range(4)},
      by_last={str(k):measures(c[[ORDERS[i%6][-1]==k for i in ix]],y[[ORDERS[i%6][-1]==k for i in ix]]) for k in range(3)},
      same_last=dict(pairs=len(pairs),both_strict_correct=sum(bool(pred[a]==y[a] and pred[b]==y[b] and not tie[a] and not tie[b]) for a,b in pairs),raw_distinct=sum(not np.array_equal(c[a],c[b]) for a,b in pairs)))

def acquired(pre,post):return post+1e-12>=CRITERIA['accuracy'] and post-pre+1e-12>=CRITERIA['gain']

def contrast_pass(mean,sign):return mean+1e-12>=CRITERIA['contrast'] and sign['positive']>sign['negative'] and sign['p_two_sided']<=CRITERIA['sign_p']

def retention_pass(pre,post,lost):return all(pre[k]-post[k]<=CRITERIA['retention_drop']+1e-12 for k in ('A','B')) and lost<=CRITERIA['lost_pairs']

def acquisition_pass(count,contrasts):return count>=CRITERIA['count'] and all(c['pass_contrast'] for c in contrasts.values())

def readout(p,x,cut=None,old=False):
    n=N.clone(p);n.reset(len(x))
    for k in ('w','theta','neighbors'):getattr(n,k).flags.writeable=False
    masks=[];c0=None;qm=None
    for t in range(x.shape[1]):
        if t==cut:n.reset(len(x))
        if not old and t==64:c0=n.count.copy()
        prev=n.count.copy();n.step(x[:,t],None,False)
        if old or t>=64:masks.append(((n.count-prev).astype(np.int64)*(1<<np.arange(16))).sum(-1))
        if t==80:qm=n.count.copy()-c0 if c0 is not None else None
    counts=n.count if old else n.count-c0
    if n.payload()!=p or n.teacher_dose or n.dw_ticks:raise ValueError('evaluation contamination')
    mm=np.array(masks,dtype=np.int64).T
    return dict(counts=counts.astype(int).tolist(),Q=None if old else qm.astype(int).tolist(),raster_sha256=[tensor_sha(a) for a in mm],payload_sha256=digest(p))

def evaluate(p,xtrain,xtest,xold,ytrain,ytest,oldlabels):
    rows={}
    for k,x,y in [('train',xtrain,ytrain),('test',xtest,ytest),('old',xold,oldlabels)]:
        r=readout(p,x,old=k=='old');r['metrics']=measures(r['counts'],y)
        if k=='old':r['tasks']={name:measures(np.array(r['counts'])[np.arange(len(y))%4//2==j],np.array(y)[np.arange(len(y))%4//2==j]) for j,name in enumerate(('A','B'))}
        else:
            r.update(strata(r['counts'],y))
        rows[k]=r
    return rows

def resets_check(p,xtest):
    out={}
    for cut in (44,64):
        r=readout(p,xtest,cut=cut);c=np.array(r['counts'])
        if cut==44:
            if any(not np.array_equal(c[a],c[b]) for a,b in pairs()):raise ValueError('last reset mismatch')
        elif any(not np.all(c[g*6:g*6+6]==c[g*6]) for g in range(40)):raise ValueError('Q reset mismatch')
        out[str(cut)]=r
    return out

def summarize(out):
    values={a+'/'+p:[] for a in ARMS for p in POLICIES}
    for seed in SEEDS:
        d=panel(seed);data_check(d);r=load(out/f'result_{seed}.json')
        if r['seed']!=seed or r['data_sha256']!=digest(d):raise ValueError('result identity')
        if any(set(r[k])!=set(values) for k in ('rows','checkpoints','journals','restores','resets')):raise ValueError('result inventory')
        for key in values:
            rr=r['rows'][key];cc=r['checkpoints'][key];jj=r['journals'][key];arm,pol=key.split('/')
            if cc['0']!=checkpoint(seed,arm) or set(cc)!={str(i) for i in POINTS} or len(jj)!=96:raise ValueError('checkpoint inventory')
            before=digest(cc['0'])
            for j,(i,v) in enumerate(zip(d['order'],jj),1):
                expected_label=d['teacher_labels'][i] if pol=='label_shuffle' else LABELS[i%6]
                expected_start=128 if pol=='delayed' else d['time_starts'][i] if pol=='time_shuffle' else 64
                if v['before']!=before or v['start']!=expected_start or v['label']!=expected_label or abs(v['teacher_dose']-115.2)>1e-8:raise ValueError('training journal')
                diag=('teacher_dose','teacher_spikes','preview_spikes','ca_peak','ltp_slots','ltd_slots','weight_l1','theta_l1','dw_slots','clipped')
                if any(not isinstance(v[k],(int,float)) or not math.isfinite(v[k]) or v[k]<0 for k in diag):raise ValueError('journal numeric domain')
                if pol=='frozen' and any(v[k]!=0 for k in ('weight_l1','theta_l1','dw_slots','clipped')):raise ValueError('frozen diagnostics')
                before=v['after']
                if j in POINTS and before!=digest(cc[str(j)]):raise ValueError('checkpoint chain')
            if pol=='frozen' and any(p!=cc['0'] for p in cc.values()):raise ValueError('frozen changed')
            if r['restores'][key] is not True or set(r['resets'][key])!={'44','64'}:raise ValueError('restore/reset inventory')
            for cut,row in r['resets'][key].items():
                validate_readout(row,cc['96'],240)
                c=np.array(row['counts'])
                if cut=='44' and any(not np.array_equal(c[a],c[b]) for a,b in pairs()):raise ValueError('last reset mismatch')
                if cut=='64' and any(not np.all(c[g*6:g*6+6]==c[g*6]) for g in range(40)):raise ValueError('Q reset mismatch')
            trajectory=[]
            for step in POINTS:
                pt=str(step);N.clone(cc[pt])
                for split,y in [('train',np.tile(LABELS,16)),('test',np.tile(LABELS,40)),('old',np.tile([0,1,0,1],10))]:
                    row=rr[pt][split]
                    validate_readout(row,cc[pt],len(y),old=split=='old',enriched=True)
                    if row['metrics']!=measures(row['counts'],y) or row['payload_sha256']!=digest(cc[pt]):raise ValueError('evaluation metric')
                    if split!='old' and any(row[k]!=v for k,v in strata(row['counts'],y).items()):raise ValueError('strata metric')
                for j0,t in enumerate(('A','B')):
                    ix=np.arange(40)%4//2==j0
                    if rr[pt]['old']['tasks'][t]!=measures(np.array(rr[pt]['old']['counts'])[ix],np.tile([0,1,0,1],10)[ix]):raise ValueError('retention metric')
                tasks={t:rr[pt]['old']['tasks'][t]['neutral_correct']/20 for t in ('A','B')}
                trajectory.append(dict(point=step,accuracy=rr[pt]['test']['metrics']['neutral_correct']/240,train_accuracy=rr[pt]['train']['metrics']['neutral_correct']/96,**tasks))
            pre,post=trajectory[0],trajectory[-1]
            values[key].append(dict(seed=seed,trajectory=trajectory,acquired=acquired(pre['accuracy'],post['accuracy']),
              lost=sum(pre[t]>=.8 and post[t]<.8 for t in ('A','B')),prior_unacquired=sum(pre[t]<.8 for t in ('A','B')),
              newly_acquired=sum(pre[t]<.8 and post[t]>=.8 for t in ('A','B')),never_acquired=sum(all(v[t]<.8 for v in trajectory) for t in ('A','B')),
              recovered=sum(pre[t]>=.8 and post[t]>=.8 and any(v[t]<.8 for v in trajectory[1:-1]) for t in ('A','B')),
              W_change=float(np.abs(np.array(cc['96']['weights'])-np.array(cc['0']['weights'])).sum()),theta_change=float(np.abs(np.array(cc['96']['thresholds'])-np.array(cc['0']['thresholds'])).sum())))
    totals={}
    for key,vs in values.items():
        totals[key]=dict(acquired=sum(v['acquired'] for v in vs),lost=sum(v['lost'] for v in vs),prior_unacquired=sum(v['prior_unacquired'] for v in vs),newly_acquired=sum(v['newly_acquired'] for v in vs),never_acquired=sum(v['never_acquired'] for v in vs),recovered=sum(v['recovered'] for v in vs),
          trajectory=[{k:sum(v['trajectory'][i][k] for v in vs)/30 for k in ('accuracy','train_accuracy','A','B')} for i in range(4)],W_change=sum(v['W_change'] for v in vs)/30,theta_change=sum(v['theta_change'] for v in vs)/30)
    contrasts={}
    for pol in ('frozen','label_shuffle'):
        delta=[a['trajectory'][-1]['accuracy']-b['trajectory'][-1]['accuracy'] for a,b in zip(values['local/immediate'],values['local/'+pol])]
        sign=sign_test(delta);contrasts[pol]=dict(mean=sum(delta)/30,sign=sign,pass_contrast=contrast_pass(sum(delta)/30,sign))
    t=totals['local/immediate'];acq=acquisition_pass(t['acquired'],contrasts)
    retention=retention_pass(t['trajectory'][0],t['trajectory'][-1],t['lost'])
    return dict(values=values,totals=totals,contrasts=contrasts,acquisition_pass=acq,retention_pass=retention,joint_pass=acq and retention)
