"""Connect provisional pair evidence upstream of the SAME physical release gate.
Does not import or launch a network. No target, slot, clock, or permission bit in tick()."""
from contextlib import contextmanager
import numpy as np
from .association import Durable,Episode,need
class Readout:
    def __init__(self,choice,weights,lr,w):
        need(isinstance(weights,Durable),'invalid contact payload')
        self.c=choice;self.weights=weights;self.rows=[Episode(lr,w) for _ in choice.last]
        self.prepared=np.zeros(len(self.rows),np.int8)
        ww=choice.w;ix=np.flatnonzero(ww)
        from itertools import combinations
        need(len(ix)==1 and list(combinations(range(8),2))[int(ix[0])]==tuple(lr) and ww[ix[0]]==w and choice.norm==1,'projection provenance')
        need(not choice.last.any() and choice.last.shape==(len(self.rows),8),'initialization')
    def episode_credit(self,i,feedback):
        # External teacher supplies only +/- direction, after the episode. No address/slot from the scorer.
        need(type(i) is int and 0<=i<len(self.rows),'invalid stream')
        cu=int(self.c.eligible[i])
        if not cu or not np.any(self.c.mem.keys==cu):return ()
        m=self.c.mem.mixture(self.c.eligible[i:i+1])[0]
        return self.rows[i].credit_proposal(cu,m,feedback)
    @contextmanager
    def attached(self):
        c=self.c;n=c.n;prev=n.step;had='step' in vars(n);saved=vars(n).get('step')
        pending=ids=None
        def callback(x,current,learn):
            nonlocal pending,ids
            need(learn is False and pending is not None and np.array_equal(x,pending),'nonphysical callback')
            need(current.shape==(len(self.rows),16),'drive shape')
            pending=None
            if len(ids):
                mix=c.mem.mixture(c.eligible[ids])
                for i,m in zip(ids,mix):
                    cu=int(c.eligible[i]);self.prepared[i]=self.rows[i].read(self.weights,cu,m)
            # Cast before scaling, with exactly the old maximum magnitude on each somatic group.
            bias=1.8*self.prepared*(x>0).any(1)
            out=np.empty_like(current);out[:,:8]=-bias[:,None];out[:,8:]=bias[:,None]
            return prev(x,out,False)
        def tick(x):
            nonlocal pending,ids
            need(pending is None and x.shape==c.last.shape and np.isfinite(x).all() and (x>=0).all(),'bad tick')
            on=x>0;ids=np.flatnonzero((on&~c.last).any(1));pending=x
            if len(ids):
                codes=on[ids]@(1<<np.arange(8))
                for i,k in zip(ids,codes):self.rows[i].observe(int(k))
            c.tick(x);need(pending is None,'downstream missing')
        n.step=callback
        try:yield tick
        finally:
            if had:n.step=saved
            else:delattr(n,'step')
