"""N2 dimension-generalized N1.8 dynamics, sparse inhibitory routing.

All constants phenomenological. No human-calibrated claim. Eight afferent
weights/cell, two dendrites and two incoming inhibitory edges/cell. Event
routing is exact integer delivery, NOT skipping silent cellular timesteps.
"""
import json, hashlib
import numpy as np
from .paths import ROOT
C=json.loads((ROOT/'configs/cell.json').read_text())
FAST=('p','d','s','bp','ca','count','a','inhibition')

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def graph_check(g,n):
    g=np.asarray(g)
    if g.shape!=(n,2) or not np.issubdtype(g.dtype,np.integer) or (g<0).any() or (g>=n).any() or (g==np.arange(n)[:,None]).any() or (g[:,0]==g[:,1]).any():raise ValueError('invalid sparse graph')
def ring(n):return np.stack(((np.arange(n)-1)%n,(np.arange(n)+1)%n),axis=1).astype(np.int64)
def feedback(theta,counts):return np.clip(theta+.005*(np.minimum(counts,4)-1),.15,.65)

class Network:
    def __init__(self,n,seed,policy='local',routing='pull',graph=None):
        if type(n)!=int or n<4 or n%2 or n>16000 or policy not in ('local','baseline') or routing not in ('pull','event'):raise ValueError('identity')
        self.n,self.seed,self.policy,self.routing=n,seed,policy,routing;self.c=dict(C,cells=n)
        self.w=np.random.default_rng(seed+100000).uniform(C['init_low'],C['init_high'],(n,8));self.theta=np.full(n,.4)
        self.set_graph(ring(n) if graph is None else graph);self.reset()
    def set_graph(self,g):
        graph_check(g,self.n);self.neighbors=np.array(g,dtype=np.int64,copy=True)
        # Reverse CSR: sorted source IDs; stable target/slot ordering.
        sources=self.neighbors.ravel();order=np.argsort(sources,kind='stable')
        self.targets=np.repeat(np.arange(self.n,dtype=np.int64),2)[order]
        self.ptr=np.r_[0,np.cumsum(np.bincount(sources,minlength=self.n))].astype(np.int64)
    def reset(self,batch=1):
        self.p=np.zeros((batch,8));self.d=np.zeros((batch,self.n,2));self.s=np.zeros((batch,self.n));self.bp=np.zeros_like(self.s)
        self.ca=np.zeros((batch,self.n,8));self.count=np.zeros_like(self.s);self.a=np.zeros_like(self.s);self.inhibition=np.zeros_like(self.s)
        self.work=dict(steps=0,cell_updates=0,calcium_slots=0,plastic_slots=0,input_synapse_products=0,pull_edge_reads=0,event_source_scans=0,event_deliveries=0,teacher_dose=0.,teacher_spikes=0,logical_state_rw_bytes=0)
    def route(self,spike):
        if self.routing=='pull':
            self.work['pull_edge_reads']+=spike.shape[0]*2*self.n
            return spike[:,self.neighbors].sum(-1)
        out=np.zeros(spike.shape,dtype=np.int64)
        self.work['event_source_scans']+=spike.size
        for b,z in enumerate(spike):
            active=np.flatnonzero(z);deg=self.ptr[active+1]-self.ptr[active];total=int(deg.sum())
            if total:
                offset=self.ptr[active]-np.cumsum(deg)+deg
                edges=np.repeat(offset,deg)+np.arange(total)
                out[b]=np.bincount(self.targets[edges],minlength=self.n)
            self.work['event_deliveries']+=total
        return out
    def step(self,x,teacher=None,learn=False,nmda=True,vdcc=True):
        c=self.c;dt=c['dt'];batch=self.s.shape[0];x=np.asarray(x).reshape(batch,8)
        if learn and batch!=1:raise ValueError('sequential learning only')
        original=np.zeros_like(self.s) if teacher is None else np.asarray(teacher)
        if original.shape!=self.s.shape:raise ValueError('teacher shape')
        supplied=(original-self.inhibition if self.policy=='local' else original)-self.a
        self.p=self.p*np.exp(-dt/c['tau_pre'])+x*(-np.expm1(-dt/c['tau_pre']))
        current=(self.p[:,None,:]*self.w[None,:,:]).reshape(batch,self.n,2,4).sum(-1)
        old_d,old_s=self.d,self.s
        self.d=old_d+dt/c['tau_d']*(-old_d+c['coupling_d']*(old_s[:,:,None]-old_d)+current+self.bp[:,:,None])
        next_s=old_s+dt/c['tau_s']*(-old_s+c['coupling_s']*(old_d.mean(-1)-old_s)+supplied)
        z=next_s>=self.theta;self.s=np.where(z,0.,next_s)
        self.bp=self.bp*np.exp(-dt/c['tau_bp'])+c['bp_jump']*z;self.count+=z
        voltage=np.repeat(self.d,4,axis=-1);ng=np.clip((voltage-c['nmda_offset'])/c['nmda_scale'],0,1);vg=np.clip((voltage-c['vdcc_offset'])/c['vdcc_scale'],0,1)
        source=int(nmda)*c['nmda_gain']*self.p[:,None,:]*ng+int(vdcc)*c['vdcc_gain']*vg
        decay=np.exp(-dt/c['tau_ca']);self.ca=self.ca*decay+source*c['tau_ca']*(1-decay)
        if learn:
            used=self.ca[0];high=used>=c['theta_p'];active=used>=c['theta_d']
            rate=np.where(high,c['rate_p'],np.where(active,c['rate_d'],0.));target=np.where(high,c['w_high'],c['w_low'])
            alpha=-np.expm1(-dt*rate*self.p[0]);self.w=self.w+alpha*(target-self.w)
        self.a=self.a*np.exp(-dt/c['tau_adapt'])+c['adapt_jump']*z
        if self.policy=='local':self.inhibition=self.inhibition*np.exp(-dt/2.)+.1*self.route(z)
        w=self.work;w['steps']+=1;w['cell_updates']+=batch*self.n;w['calcium_slots']+=8*batch*self.n;w['input_synapse_products']+=8*batch*self.n;w['plastic_slots']+=8*self.n if learn else 0
        w['teacher_dose']+=float(original.sum()*dt);w['teacher_spikes']+=int(z[original>0].sum())
        # Explicit logical accounting, not cache misses/DRAM traffic.
        w['logical_state_rw_bytes']+=2*sum(getattr(self,k).nbytes for k in FAST)+self.w.nbytes+(self.w.nbytes if learn else 0)
        return z
    def payload(self):
        return dict(kind='N2.0',n=self.n,seed=self.seed,policy=self.policy,config_sha256=digest(C),weights=self.w.tolist(),thresholds=self.theta.tolist(),neighbors=self.neighbors.tolist())
    def restore(self,p):
        expected={'kind','n','seed','policy','config_sha256','weights','thresholds','neighbors'}
        if set(p)!=expected or p['kind']!='N2.0' or p['n']!=self.n or p['seed']!=self.seed or p['policy']!=self.policy or p['config_sha256']!=digest(C):raise ValueError('payload identity')
        w,t=np.asarray(p['weights'],dtype=np.float64),np.asarray(p['thresholds'],dtype=np.float64)
        if w.shape!=(self.n,8) or t.shape!=(self.n,) or not np.isfinite(w).all() or not np.isfinite(t).all() or (w<.05).any() or (w>1).any() or (t<.15).any() or (t>.65).any():raise ValueError('payload domain')
        # Validate ALL input before touching this live object. Stage graph/reset
        # on a separate object so invalid checkpoints cannot partly mutate it.
        graph_check(p['neighbors'],self.n)
        staged=Network(self.n,self.seed,self.policy,self.routing,p['neighbors'])
        staged.w=w.copy();staged.theta=t.copy()
        self.__dict__.update(staged.__dict__)
    def bytes(self):return dict(learned=self.w.nbytes+self.theta.nbytes,graph=self.neighbors.nbytes,reverse_index=self.targets.nbytes+self.ptr.nbytes,fast=sum(getattr(self,k).nbytes for k in FAST))
    def check(self):
        if not all(np.isfinite(getattr(self,k)).all() for k in FAST+('w','theta')):raise ValueError('nonfinite state')
        if (self.w<.05).any() or (self.w>1).any():raise ValueError('weight bound')

def clone(p,routing='pull'):
    n=Network(p['n'],p['seed'],p['policy'],routing);n.restore(p);return n
