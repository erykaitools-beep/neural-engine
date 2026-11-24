import json, hashlib
import numpy as np
from . import sparse as C
from .paths import ROOT
def config(): return dict(C.C)
def stimulus(example, c=None):
    c = c or config()
    t = np.arange(round(c['duration']/c['dt'])) * c['dt']
    x = np.zeros((len(t), 8))
    gate = (t >= 6+example['jitter']) & (t < 14+example['jitter'])
    x[np.ix_(gate, example['channels'])] = example['amplitude']
    return x
def tensor_sha(x):return hashlib.sha256(np.asarray(x,dtype='<f8').tobytes()).hexdigest()
def dataset(seed):
    templates=json.loads((ROOT/'configs/templates.json').read_text());trng=np.random.default_rng(seed+1000);erng=np.random.default_rng(seed+2000)
    def sample(t,rng):return dict(t,amplitude=float(rng.uniform(.8,1.2)),jitter=float(rng.uniform(-.5,.5)))
    train=[]
    for task in (0,1):
        rows=[sample(templates[task*2+j],trng) for _ in range(32) for j in (0,1)]
        train.extend([rows[i] for i in trng.permutation(64)])
    test=[sample(t,erng) for _ in range(8) for t in templates]
    if {tensor_sha(stimulus(e)) for e in train}&{tensor_sha(stimulus(e)) for e in test}:raise ValueError('split collision')
    return dict(seed=seed,train=train,test=test)

def metrics(counts,labels,n):
    c=np.asarray(counts);y=np.asarray(labels);scores=c.reshape(-1,2,n//2).sum(-1);ties=scores[:,0]==scores[:,1];pred=scores.argmax(-1)
    return dict(accuracy=float(np.mean((pred==y)&~ties)+.5*np.mean(ties)),ties=int(ties.sum()),silent=int((c.sum(-1)==0).sum()),spikes=int(c.sum()))

def readout(p,data,clock):
    net=C.clone(p);x=np.stack([stimulus(e) for e in data]);net.reset(len(x))
    for k in ('w','theta','neighbors'):getattr(net,k).flags.writeable=False
    for t in range(80):net.step(x[:,t]);clock.tick()
    if net.payload()!=p:raise ValueError('readout changed state')
    counts=net.count.astype(int).tolist();labels=[e['label'] for e in data]
    tasks={}
    for task in ('A','B'):
        ids=[i for i,e in enumerate(data) if e['phase']==task];tasks[task]=metrics(np.array(counts)[ids],np.array(labels)[ids],net.n)
    return dict(counts=counts,tasks=tasks,payload_sha256=C.digest(p),work=net.work)

def train(net,e,clock):
    x=stimulus(e);net.reset();before=net.w.copy();theta=net.theta.copy();work=[]
    for xx in x:net.step(xx);clock.tick()
    preview=net.count[0].copy();work.append(dict(net.work));net.theta=C.feedback(net.theta,preview);net.reset()
    teacher=np.zeros((1,net.n));y=e['label'];teacher[:,y*(net.n//2):(y+1)*(net.n//2)]=1.8
    for t,xx in enumerate(x):net.step(xx,teacher if 16<=t<32 else None,True);clock.tick()
    net.check();work.append(dict(net.work))
    journal=dict(preview_counts=preview.astype(int).tolist(),W_delta=float(np.abs(net.w-before).sum()),theta_delta=float(np.abs(net.theta-theta).sum()),work=work)
    net.reset();return preview,journal

def validate_work(w,n,batch=1,learn=False):
    keys=set(C.Network(4,0).work)
    if set(w)!=keys or any(not isinstance(v,(int,float)) or not np.isfinite(v) or v<0 for v in w.values()):raise ValueError('work domain')
    expected=dict(steps=80,cell_updates=80*n*batch,calcium_slots=80*8*n*batch,input_synapse_products=80*8*n*batch,plastic_slots=80*8*n if learn else 0,pull_edge_reads=80*2*n*batch,event_source_scans=0,event_deliveries=0,
      logical_state_rw_bytes=80*(2*(64+120*n)*batch+64*n+(64*n if learn else 0)))
    if any(w[k]!=v for k,v in expected.items()) or abs(w['teacher_dose']-(n/2*1.8*8 if learn else 0))>1e-6 or (not learn and w['teacher_spikes']!=0):raise ValueError('work expected')

def validate_journal(v,n):
    if set(v)!={'preview_counts','W_delta','theta_delta','work'} or len(v['work'])!=2:raise ValueError('journal inventory')
    p=np.asarray(v['preview_counts'])
    if p.shape!=(n,) or not np.issubdtype(p.dtype,np.integer) or (p<0).any() or (p>80).any():raise ValueError('preview domain')
    if any(not np.isfinite(v[k]) or v[k]<0 for k in ('W_delta','theta_delta')):raise ValueError('journal domain')
    for w,learn in zip(v['work'],(False,True)):validate_work(w,n,learn=learn)
