import numpy as np
from .integrity import need
from .context import Memory, Choice, BITS
BIT16=2**np.arange(16,dtype=np.int32)
class FrozenMemory(Memory):
    def update(self,*a):raise ValueError('forbidden training')

class Recognized(Choice):
    """Change only the post-event write to an existing uint8 address, not the soma current."""
    def tick(self,x):
        prev=self.prior.copy()
        super().tick(x)
        # At end of the original tick, prior holds the proposed mask, eligible the older one.
        known=(self.prior==self.mem.keys[0])|(self.prior==self.mem.keys[1])
        if self.arm=='first_only':known &= prev==0
        np.copyto(self.prior,prev,where=~known)

class Release:
    def __init__(self,weights=None):
        self.w=np.zeros(256,np.int16) if weights is None else np.array(weights,dtype=np.int16)
    def permits(self,row):return self.w[masks(row)]>0

def compress16(z):return (z.astype(np.int32)@BIT16).astype(np.uint16)

def masks(x):return (x>0)@BITS

class Temporal(Release):
    def __init__(self,w,memory,n):
        super().__init__(w);self.memory=memory
        self.key=np.zeros(n,np.uint8);self.run=np.zeros(n,np.uint8)
        if memory:self.prev=np.zeros(n,np.float64)
    def permits(self,row):
        mx=row.max(1);strong=row>mx[:,None]/32.
        addr=(strong@BITS).astype(np.uint8);ok=self.w[addr]>0
        cont=ok & (self.key==addr)
        if self.memory:
            small=np.where(strong,0.,row).sum(1)/np.where(mx>0,mx,1.)
            cont &= (((self.prev==0)&(small==0)) | ((self.prev>0)&(small>0)&(np.abs(small-self.prev)>1/512.)))
            self.prev[:]=small
        self.run[:]=np.where(cont,np.minimum(self.run+1,2),0)
        self.key[:]=np.where(ok,addr,0)
        return ok & (self.run==2)

def simulate(cp,payload,gate,x):
    need(x.ndim==3 and x.shape[0]==48 and x.shape[2]==8,'actual native dispatch width must be 48')
    mm=FrozenMemory(payload);mm.keys.flags.writeable=False;mm.val.flags.writeable=False
    o=Recognized(cp,'selective',mm);o.reset(len(x));o.ring.flags.writeable=False;o.w.flags.writeable=False
    n=o.n;native=n.step;reg=[None,None,None]
    def physical(row,current,learn):
        need(learn is False,'unexpected core learning');allow=np.ones(len(row),bool) if gate is None else gate.permits(row)
        need(row.shape==(48,8) and current.shape==(48,16),'actual tick extent')
        delivered=current*allow[:,None];reg[:]=[current[:,8],delivered[:,8],allow]
        return native(row,delivered,False)
    n.step=physical;b,t=x.shape[:2];bits=np.empty((b,t),np.uint16);drive=np.empty((b,t,2),np.float64);enabled=np.empty((b,t),np.uint8)
    for i,row in enumerate(x.transpose(1,0,2)):
        count=n.count.copy();o.tick(row);sp=n.count-count
        need(np.all((sp==0)|(sp==1)),'spike encoding');bits[:,i]=compress16(sp)
        drive[:,i,0],drive[:,i,1],enabled[:,i]=reg
    need(n.payload()==cp['core'] and o.payload()==cp['aux'] and mm.payload()==payload and not o.writes and not n.dw_slots,'frozen core drift')
    return bits,drive,enabled

def fixtures_t():
    w=np.zeros(256,np.int16);w[7]=1
    def probe(vals,mem):
        gate=Temporal(w,mem,2);out=[]
        for a,b in vals:
            x=np.zeros((2,8))
            for i,z in enumerate((a,b)):
                if z is not None:x[i,:3]=1.;x[i,3]=z
            out.append(gate.permits(x).tolist())
        return np.array(out).T.tolist()
    pat=[(.01,0),(.02,None),(.01,0),(.01,0),(.02,0),(.01,None),(None,0),(.02,None)]
    need(probe(pat,True)==[[False,False,True,False,False,True,False,False],[False,False,False,False,True,False,False,False]],'reset, weak history, row independence')
    need(probe(pat,False)[0]==[False,False,True,True,True,True,False,False],'delay not matched')
    for diff,verdict in [(1/1024.,False),(1/512.,False),(1/256.,True)]:
        a=1/128.;need(probe([(a,0),(a+diff,0),(a,0)],True)[0][-1]==verdict,'change threshold')
class Contact:
    """Two mutable address contacts, two order-specific eligibility counters apiece."""
    def __init__(self,p=None,n=48,early=False,shared=False):
        self.keys=np.zeros(2,np.uint8);self.weights=np.zeros((2,2),np.int16)
        if p is not None:self.keys[:]=p['keys'];self.weights[:]=p['weights']
        self.early=early;self.shared=shared
        self.addr=np.zeros(n,np.uint8);self.prev=np.zeros(n,np.uint8);self.run=np.zeros(n,np.uint8)
        self.first=np.zeros(n,np.uint8);self.both=np.zeros(n,bool)
    def data(self):return dict(keys=self.keys.tolist(),weights=self.weights.tolist())
    def permits(self,row):
        mx=row.max(1);strong=row>mx[:,None]/32.;k=(strong@BITS).astype(np.uint8)
        residual=np.where(strong,0.,row).sum(1)/np.where(mx>0,mx,1.)
        lev=np.where(residual>1/80.,2,np.where(residual>0,1,0)).astype(np.uint8)
        active=lev!=0;reset=(k!=self.addr)|~active
        self.first[reset]=0;self.both[reset]=False
        self.run[:]=np.where(~active,0,np.where(reset|(lev!=self.prev),1,np.minimum(3,self.run+1)))
        self.prev[:]=lev;self.addr[:]=np.where(active,k,0)
        event=self.run==3
        self.both |= event&(self.first!=0)&(self.first!=lev)
        self.first[:]=np.where(event&(self.first==0),lev,self.first)
        permit=np.zeros(len(row),bool)
        for i in range(2):
            permit |= active&(k==self.keys[i])&(self.keys[i]!=0)&(self.first!=0)&(self.both|self.early)&(self.weights[i,(self.first==2).astype(int)]>0)
        return permit
    def teach(self,target):
        need(len(self.addr)==1 and self.both[0] and self.first[0] in (1,2) and target in (-1,1),'incomplete eligibility')
        k=int(self.addr[0]);need(k!=0,'no physical address');idx=np.flatnonzero(self.keys==k)
        i=int(idx[0]) if len(idx) else int(np.flatnonzero(self.keys==0)[0]);self.keys[i]=k
        if self.shared:self.weights[:,int(self.first[0]==2)]+=target
        else:self.weights[i,int(self.first[0]==2)]+=target

class Paired(Contact):
    def __init__(self,cp,n,mode):
        super().__init__(n=n);self.table=cp;self.mode=mode;self.older=np.zeros(n,np.uint8);self.recent=self.older.copy()
    def permits(self,row):
        super().permits(row)
        pure=(self.addr==0);on=(row>0);code=on@BITS
        write=pure&(on.sum(1)==3)&(code!=self.recent)
        self.older[write]=self.recent[write];self.recent[write]=code[write]
        vote=np.zeros(len(row),np.int16);ix=(self.first==2).astype(int)
        for kk,ww in zip(self.table['keys'],self.table['weights']):
            if not kk:continue
            seen=(self.recent==kk) if self.mode=='last' else (self.older==kk) if self.mode=='penult' else ((self.recent==kk)|(self.older==kk))
            vote+=np.where(seen,np.asarray(ww,np.int16)[ix],0).astype(np.int16)
        return self.both&(self.addr!=0)&(vote>0)

class Filter(Recognized):
    def setup(self,n,policy):
        self.policy=policy;self.writes_seen=np.zeros(n,np.uint16);self.writes_kept=self.writes_seen.copy()
        if policy=='repeat':self.r17_seen_support_bits=np.zeros((n,32),np.uint8)
    def tick(self,row):
        on=row>0;idx=np.flatnonzero((on&~self.last).any(1))
        if not len(idx):return super().tick(row)
        self.writes_seen[idx]+=1
        if self.policy=='repeat':
            keys=(on[idx]@BITS).astype(int);slot=keys//8;bit=(1<<(keys%8)).astype(np.uint8)
            ok=(self.r17_seen_support_bits[idx,slot]&bit)!=0;self.r17_seen_support_bits[idx,slot]|=bit
        elif self.policy=='ordinal':ok=(self.writes_seen[idx]==3)|(self.writes_seen[idx]==4)
        else:need(self.policy=='all','unknown writer');ok=np.ones(len(idx),bool)
        self.writes_kept[idx]+=ok;deny=idx[~ok];prev=self.event[deny].copy()
        super().tick(row);self.event[deny]=prev
