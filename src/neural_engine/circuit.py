import numpy as np
from . import homeostasis as E
from .sparse import digest
ARMS=['baseline','local','global','shuffled']
RULE=dict(gain=.1,tau=2.,neighbors=2,permutation_seed_offset=810000)
CONFIG_ID=digest(dict(parent=E.CONFIG_ID,rule=RULE,version='N1.8-1'))

def topology(w,seed,policy):
    w=np.asarray(w)
    if w.shape!=(16,8) or not np.isfinite(w).all():raise ValueError('topology weights')
    distance=((w[:,None,:]-w[None,:,:])**2).sum(-1)
    np.fill_diagonal(distance,np.inf)
    neighbors=np.argsort(distance,axis=1,kind='stable')[:,:2]
    if policy=='shuffled':
        # relabel vertices, preserving graph isomorphism (not per-cell outdegree).
        perm=np.random.default_rng(seed+RULE['permutation_seed_offset']).permutation(16)
        relabeled=np.empty_like(neighbors);relabeled[perm]=perm[neighbors];neighbors=relabeled
    validate_graph(neighbors)
    neighbors.flags.writeable=False
    return neighbors

def validate_graph(n):
    n=np.asarray(n)
    if n.shape!=(16,2) or not np.issubdtype(n.dtype,np.integer) or (n<0).any() or (n>15).any():raise ValueError('graph dimensions')
    if (n==np.arange(16)[:,None]).any() or (n[:,0]==n[:,1]).any():raise ValueError('self or duplicate edge')

def drive(spikes,neighbors,policy):
    raw=spikes[:,neighbors].sum(-1)
    if policy=='global':raw=np.broadcast_to(raw.mean(1,keepdims=True),raw.shape)
    return RULE['gain']*raw

class Network(E.Network):
    def __init__(self,seed,policy='baseline'):
        if type(seed)!=int or policy not in ARMS:raise ValueError('identity')
        self.seed,self.policy=seed,policy
        super().__init__(seed,'local')
        self.neighbors=topology(self.w,seed,policy)
    def reset(self,batch=1):
        super().reset(batch)
        self.inhibition=np.zeros_like(self.s)
        self.inhib_dose=0.;self.inhib_peak=0.;self.dw_abs=0.;self.dw_ticks=0;self.dw_slots=0
        self.nmda_saturated=0;self.vdcc_saturated=0;self.steps=0
    def step(self,x,teacher=None,learn=False,nmda=True,vdcc=True):
        original=np.zeros_like(self.s) if teacher is None else np.asarray(teacher)
        old_count=self.count.copy();old_w=self.w.copy() if learn else None
        dose,teacher_spikes=self.teacher_dose,self.teacher_spikes
        if self.policy=='baseline':
            used=super().step(x,teacher,learn,nmda,vdcc)
        else:
            old_i=self.inhibition
            used=super().step(x,original-old_i,learn,nmda,vdcc)
            spikes=self.count-old_count
            # Undo only attribution counters; external teacher itself is unchanged.
            self.teacher_dose=dose+float(original.sum()*self.c['dt'])
            self.teacher_spikes=teacher_spikes+int(spikes[original>0].sum())
            self.inhib_dose+=float(old_i.sum()*self.c['dt'])
            self.inhibition=old_i*np.exp(-self.c['dt']/RULE['tau'])+drive(spikes,self.neighbors,self.policy)
            self.inhib_peak=max(self.inhib_peak,float(self.inhibition.max()))
            if not np.isfinite(self.inhibition).all() or (self.inhibition<0).any():raise FloatingPointError('inhibition state')
        if learn:
            delta=self.w-old_w
            self.dw_abs+=float(np.abs(delta).sum());self.dw_ticks+=int(np.any(delta));self.dw_slots+=int(np.count_nonzero(delta))
        self.nmda_saturated+=int(np.count_nonzero(self.d>=self.c['nmda_offset']+self.c['nmda_scale']))*4
        self.vdcc_saturated+=int(np.count_nonzero(self.d>=self.c['vdcc_offset']+self.c['vdcc_scale']))*4
        self.steps+=1
        return used
    def diagnostic(self):
        return dict(inhibition_dose=self.inhib_dose,inhibition_peak=self.inhib_peak,
                    dw_abs=self.dw_abs,dw_ticks=self.dw_ticks,dw_slots=self.dw_slots,
                    nmda_saturated_slots=self.nmda_saturated,vdcc_saturated_slots=self.vdcc_saturated,
                    steps=self.steps,teacher_dose=self.teacher_dose,teacher_spikes=self.teacher_spikes,
                    spikes=int(self.count.sum()),ca_max_endpoint=float(self.ca.max()))
    def payload(self):
        return dict(kind='N1.8',config_id=CONFIG_ID,seed=self.seed,policy=self.policy,
                    neighbors=self.neighbors.tolist(),weights=self.w.tolist(),thresholds=self.theta.tolist())
    def restore(self,p):
        if set(p)!={'kind','config_id','seed','policy','neighbors','weights','thresholds'} or p['kind']!='N1.8' or p['config_id']!=CONFIG_ID or p['seed']!=self.seed or p['policy']!=self.policy:raise ValueError('checkpoint identity')
        n=np.asarray(p['neighbors']);validate_graph(n)
        if not np.array_equal(n,self.neighbors):raise ValueError('checkpoint graph mismatch')
        E.Network.restore(self,dict(kind='N1.3',config_id=E.CONFIG_ID,mode='local',weights=p['weights'],thresholds=p['thresholds']))

def clone(p):
    n=Network(p['seed'],p['policy']);n.restore(p)
    if any(np.any(getattr(n,k)) for k in ('p','d','s','bp','ca','count','a','inhibition')):raise ValueError('restore fast state')
    return n
