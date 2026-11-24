"""Untrained candidate. No imports from the engine, generator or scorer.
Bounded sensory vocabulary, no teacher, time or target in the inference interface.
One episode per object; only Durable is serialized across episodes.
"""
from struct import Struct
from itertools import combinations
CAP=8
# headerless fixed layout in bytes; explicit little-endian/no implicit alignment
P=Struct('<BBBb')        # previously learned cue address, lower mask, upper mask, signed reliability
E=Struct('<BBbbB')      # canonical pair, first/most recent signed transition, count capped at two
def need(value,msg):
    if not value:raise ValueError(msg)
def byte(k):return type(k) is int and 0<=k<=255
class Durable:
    def __init__(self,blob=None):
        self.data=bytes(CAP*P.size) if blob is None else bytes(blob)
        need(len(self.data)==CAP*P.size,'durable size')
        keys=set()
        for cue,a,b,w in P.iter_unpack(self.data):
            if cue==0:need((a,b,w)==(0,0,0),'nonempty unused slot');continue
            need(0<a<b and -31<=w<=31 and (cue,a,b) not in keys,'invalid learned slot')
            keys.add((cue,a,b))
    def table(self):return {(c,a,b):w for c,a,b,w in P.iter_unpack(self.data) if c}
    def successor(self,changes):
        # Transactional: no in-place training, no implicit evictions or forgetting on overflow.
        vals=self.table()
        for key,inc in changes:
            need(isinstance(key,tuple) and len(key)==3 and all(byte(k) and k for k in key) and key[1]<key[2] and type(inc) is int and inc in (-1,1),'invalid update')
        need(len({k for k,_ in changes})==len(changes),'duplicate update')
        for key,inc in changes:vals[key]=max(-31,min(31,vals.get(key,0)+inc))
        need(len(vals)<=CAP,'durable capacity')
        data=bytearray(CAP*P.size)
        for j,(key,w) in enumerate(sorted(vals.items())):P.pack_into(data,P.size*j,*key,w)
        return Durable(data)
class Episode:
    def __init__(self,lr,w):
        need(tuple(lr) in combinations(range(8),2) and type(w) is int and w in (-1,1),'checkpoint projection')
        self.lr=lr;self.w=w
        self.buf=bytearray(1+CAP*E.size)      # previous mask and event-pair working memory
    def side(self,k):
        l,r=self.lr
        return int(bool(k&(1<<l)))-int(bool(k&(1<<r)))
    def observe(self,k):
        """Called for a physical support-add edge only, including after silence; no slot or class ID."""
        need(byte(k) and k!=0,'edge mask')
        prev=self.buf[0];sgn=self.side(k)*self.w
        if not prev or self.side(prev)*self.side(k)!=-1:
            self.buf[0]=k;return
        a,b=sorted((k,prev));free=None
        for j,(u,v,first,last,n) in enumerate(E.iter_unpack(memoryview(self.buf)[1:])):
            if n==0 and free is None:free=j
            if (a,b)==(u,v):
                E.pack_into(self.buf,1+j*E.size,a,b,first,sgn,min(2,n+1));self.buf[0]=k;return
        need(free is not None,'episode capacity')
        E.pack_into(self.buf,1+free*E.size,a,b,sgn,sgn,1);self.buf[0]=k
    def evidence(self,cue,mix):
        """Cue and mixture come from the existing recognized previous physical address, never the label."""
        need(byte(cue) and len(mix)==2 and all(v==v and -1<=v<=1 for v in mix) and sum(map(abs,mix))<=1+1e-14,'cue/mix')
        if cue==0:return ()
        out=[]
        for a,b,first,last,n in E.iter_unpack(memoryview(self.buf)[1:]):
            if n<2:continue
            signed=mix[0]*first+mix[1]*last
            if signed:out.append(((cue,a,b),1 if signed>0 else -1))
        return tuple(out)
    def read(self,memory,cue,mix):
        vals=memory.table()
        z=sum(max(0,vals.get(key,0))*v for key,v in self.evidence(cue,mix))
        return (z>0)-(z<0)  # tie/no evidence: no prepared drive
    def credit_proposal(self,cue,mix,post_feedback):
        """Call only AFTER readout; none => no update; scored trials must never call successor."""
        need(post_feedback is None or (type(post_feedback) is int and post_feedback in (-1,1)),'feedback')
        if post_feedback is None:return ()
        return tuple((k,post_feedback*v) for k,v in self.evidence(cue,mix))
