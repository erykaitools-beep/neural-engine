import numpy as np
from itertools import combinations
from .integrity import need
from . import circuit
IJ=np.array(list(combinations(range(8),2)));LEFT,RIGHT=IJ.T
LOOK={tuple(z):i for i,z in enumerate(IJ)}
DER=np.roll(np.arange(8),1);SEL=[];DIR=[]
for a,b in IJ:
    i,j=DER[a],DER[b];SEL.append(LOOK[tuple(sorted((i,j)))]);DIR.append(1 if i<j else -1)
DIR=np.array(DIR)
class EdgePath:
    def __init__(self,core,mode,ring=None):
        self.n=core;self.mode=mode;self.ring=np.zeros((24,28),np.int8) if ring is None else np.array(ring['ring'],dtype=np.int8);self.seen=0 if ring is None else ring['seen'];self.ev=self.dots=self.writes=0;self.charge=0.;self.real=0.;self.weights()
    def weights(self):
        q=self.ring.sum(axis=0);n=min(self.seen,24);self.w=(q.astype(float) if self.mode=='linear' else np.sign(q)*(abs(q)==n)*(n!=0));self.norm=max(1,float(np.abs(self.w).sum()))
    def payload(self):return dict(ring=self.ring.tolist(),seen=self.seen)
    def reset(self,b):
        self.n.reset(b);self.last=np.zeros((b,8),bool);self.latch=np.zeros((b,28),np.int8)
    def tick(self,x,tch=None,learn=False,aux=None):
        n=self.n;x=np.asarray(x).reshape(n.p.shape);on=(x>0);edge=np.any(on&~self.last,1);self.last=on;act=on.any(1);self.ev+=int(edge.sum())
        if np.any(edge):
            p=n.p[edge];left=p[:,LEFT];right=p[:,RIGHT];self.latch[edge]=np.sign(left-right)*(left>0)*(right>0)
        # Current is zero off-input (including post-Q). No teacher timing or task labels used in current generation.
        bias=np.zeros(len(x))
        if np.any(act) and self.mode!='mute':
            f=self.latch[act];f=f[:,SEL]*DIR if self.mode=='mis' else f
            bias[act]=f@self.w*(1.8/self.norm);self.dots+=int(act.sum())
        need(np.max(np.abs(bias))<1.800000000001,'current')
        dst=np.zeros_like(n.s);dst[:,:8]=-bias[:,None];dst[:,8:]=bias[:,None];old=n.count.copy() if tch is not None else None;dt=n.c['dt'];dose,spk=n.teacher_dose,n.teacher_spikes
        n.step(x,dst if tch is None else dst+tch,learn);self.charge+=float(np.abs(bias).sum())*16*dt;n.teacher_dose=dose+(float(tch.sum())*dt if tch is not None else 0.);n.teacher_spikes=spk+(int((n.count-old)[tch>0].sum()) if tch is not None else 0)
        if tch is not None and edge.any() and aux is not None:
            need(learn and len(x)==1 and aux in (0,1),'eligible event');i=self.seen%24;self.ring[i]=(2*aux-1)*self.latch[0];self.seen+=1;self.writes+=28;self.weights()
BITS=2**np.arange(8)
class Memory:
    def __init__(self,p=None):
        self.keys=np.zeros(2,np.uint8);self.val=np.zeros((2,2),np.int16)
        if p is not None:self.keys[:]=p['keys'];self.val[:]=p['val']
    def payload(self):return dict(keys=self.keys.tolist(),val=self.val.tolist())
    def update(self,key,v,teacher):
        need(key!=0 and teacher in (-1,1) and np.all(np.isin(v,(-1,0,1))),'invalid coincidence')
        found=np.flatnonzero(self.keys==key)
        if not len(found):
            free=np.flatnonzero(self.keys==0);need(len(free)>0,'bounded key capacity');i=free[0];self.keys[i]=key
        else:i=found[0]
        self.val[i]+=teacher*v.astype(np.int16)
    def mixture(self,keys):
        ws=np.zeros((len(keys),2))
        for k,row in zip(self.keys,self.val):
            if k:ws[keys==k]=row
        z=np.abs(ws).sum(1);empty=z==0;ws[empty]=.5
        ws[~empty]/=z[~empty,None]
        return ws

class Choice(EdgePath):
    def __init__(self,cp,arm,mem):
        self.arm=arm;self.mem=mem
        super().__init__(circuit.clone(cp['core']),'true',cp['aux'])
        for k in ('w','theta','neighbors'):getattr(self.n,k).flags.writeable=False
    def reset(self,b):
        super().reset(b);self.event=np.zeros((b,28),np.uint8);self.el=np.zeros((b,28),np.int8)
        self.prior=np.zeros(b,np.uint8);self.eligible=np.zeros(b,np.uint8);self.v=np.zeros((b,2))
    def tick(self,x):
        n=self.n;x=np.asarray(x).reshape(n.p.shape);on=x>0;edge=np.any(on&~self.last,1);self.last=on
        if np.any(edge):
            pre=n.p[edge];l=pre[:,LEFT];r=pre[:,RIGHT];proposal=np.sign(l-r)*(l>0)*(r>0)
            self.latch[edge]=np.where(self.latch[edge]!=0,self.latch[edge],proposal)
            hist=self.event[edge];self.el[edge]=(2*(hist>>2).astype(np.int8)-1)*((hist&3)==3)
            le=on[edge][:,LEFT];ri=on[edge][:,RIGHT]
            self.event[edge]=np.where(le&~ri,hist|5,np.where(ri&~le,(hist|2)&3,hist))
            self.eligible[edge]=self.prior[edge];self.prior[edge]=on[edge]@BITS
            self.v[edge]=np.column_stack((self.latch[edge]@self.w,self.el[edge]@self.w))/self.norm
        mix=self.mem.mixture(self.eligible)
        if self.arm=='hold':mix[:]=(1.,0.)
        elif self.arm=='event':mix[:]=(0.,1.)
        bias=1.8*(mix*self.v).sum(1)*on.any(1)
        need(np.isfinite(bias).all() and np.max(abs(bias))<1.80000000001,'invalid current')
        dst=np.empty_like(n.s);dst[:,:8]=-bias[:,None];dst[:,8:]=bias[:,None];n.step(x,dst,False)
