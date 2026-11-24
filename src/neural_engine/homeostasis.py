import numpy as np
from .adaptation import Network as Previous, config
from .sparse import digest
ARMS=['baseline','boost','global','local','local_frozen','mlp']
RULE=dict(initial=.4,boost=.2,low=.15,high=.65,eta=.005,target=1.,cap=4.)
CONFIG_ID=digest(dict(core=config(),rule=RULE,version='N1.3-1'))


def feedback(theta, counts, mode):
    theta,counts=np.asarray(theta,dtype=float),np.asarray(counts,dtype=float)
    if theta.shape!=(16,) or counts.shape!=(16,) or not np.isfinite(theta).all() or not np.isfinite(counts).all() or (counts<0).any():
        raise ValueError('invalid feedback input')
    delta=np.minimum(counts,RULE['cap'])-RULE['target']
    if mode=='global': delta=np.full(16,delta.mean())
    elif mode not in ('local','local_frozen'): raise ValueError('invalid feedback mode')
    proposed=theta+RULE['eta']*delta
    return np.clip(proposed,RULE['low'],RULE['high']), int(((proposed<RULE['low'])|(proposed>RULE['high'])).sum())


class Network(Previous):
    def __init__(self,seed,mode='local'):
        if mode not in ARMS or mode=='mlp': raise ValueError('invalid mode')
        self.mode=mode
        self.theta=np.full(16,RULE['boost'] if mode=='boost' else RULE['initial'])
        super().__init__(seed,arm='frozen' if mode=='local_frozen' else 'full',adaptation=True)

    def step(self,*args,**kwargs):
        # Only the threshold operand changes; the inherited core stays identical.
        scalar=self.c['spike_threshold']
        self.c['spike_threshold']=self.theta
        try: return super().step(*args,**kwargs)
        finally: self.c['spike_threshold']=scalar

    def train_current(self,x,label):
        if label not in (0,1): raise ValueError('invalid training label')
        before=digest(self.payload())
        self.episode(x)  # no label, plasticity or feedback update in preview
        counts=self.count[0].copy()
        if before!=digest(self.payload()): raise RuntimeError('preview changed persistent state')
        clipped=0
        if self.mode in ('local','global','local_frozen'):
            self.theta,clipped=feedback(self.theta,counts,self.mode)
        # episode resets all p,d,s,bp,Ca,a and counts before teacher exposure.
        _,_,diag,_=self.episode(x,label,learn=True)
        return dict(preview_spikes=int(counts.sum()),clipped_slots=clipped,
                    threshold_updates=int(self.mode in ('local','global','local_frozen')),
                    teacher_spikes=diag['teacher_spikes'],ca_max=diag['ca_max'])

    def payload(self):
        return dict(kind='N1.3',config_id=CONFIG_ID,mode=self.mode,
                    weights=self.w.tolist(),thresholds=self.theta.tolist())

    def restore(self,p):
        if set(p)!={'kind','config_id','mode','weights','thresholds'} or p['kind']!='N1.3' or p['config_id']!=CONFIG_ID or p['mode']!=self.mode:
            raise ValueError('invalid persistent checkpoint')
        w,t=np.asarray(p['weights'],float),np.asarray(p['thresholds'],float)
        if w.shape!=(16,8) or not np.isfinite(w).all() or w.min()<self.c['w_low'] or w.max()>self.c['w_high']:
            raise ValueError('invalid checkpoint weights')
        if t.shape!=(16,) or not np.isfinite(t).all() or t.min()<RULE['low'] or t.max()>RULE['high']:
            raise ValueError('invalid checkpoint thresholds')
        if self.mode in ('baseline','boost') and not np.all(t==(RULE['boost'] if self.mode=='boost' else RULE['initial'])):
            raise ValueError('fixed threshold changed')
        if self.mode=='global' and not np.all(t==t[0]): raise ValueError('global threshold not shared')
        self.w,self.theta=w.copy(),t.copy()
        self.reset()
