"""No teacher, clock, oracle or lane index in this frozen adapter."""
from .association import P,need
from .contact import Row
from .register import Register
class RepeatOnly:
    """Task-specific comparator, not a flexible alternative for all ordered bigram queries."""
    def __init__(self,alphabet):self.alphabet=alphabet;self.byte=0
    def tick(self,k):
        s=self.byte
        if k not in self.alphabet:self.byte=s&~4;return
        t=1+self.alphabet.index(k)
        if s&3==t and not s&4:
            if not s&24:s=(s&~32)|(int(t==2)<<5)
            s=(s&~64)|(int(t==2)<<6)|(1<<(2+t))
        self.byte=s&~7|4|t
    def answer(self):
        return tuple(0 if self.byte&24!=24 else -1 if self.byte&bit else 1 for bit in (32,64))

def build(mode,pair):
    need(mode in ('dual','simple','multiply'),'mode')
    slots=tuple((c,a,b) for c,a,b,w in P.iter_unpack(pair) if w>0)
    need(len(slots)==2 and len({c for c,_,_ in slots})==2 and len({m for c,a,b in slots for m in (a,b)})==4,'positive pair addresses')
    class Transfer(Row):
        def __init__(self,lr,w,blob):
            super().__init__(lr,w,blob);self.repeat_query=False;self.regs={}
            for cue,a,b in slots:
                ha,hb=w*self.side(a),w*self.side(b)
                need({ha,hb}=={-1,1},'physical orientation');alpha=(a,b) if ha>0 else (b,a)
                self.regs[cue]=RepeatOnly(alpha) if mode=='simple' else Register(*alpha)
        def observe(self,k):
            if k==5:self.repeat_query=True
            super().observe(k)
            for reg in self.regs.values():reg.tick(0);reg.tick(k)
        def read(self,memory,cue,mix):
            if not self.repeat_query:return super().read(memory,cue,mix)
            weight=self.bank.get(self.marker,0)
            if not weight or cue not in self.regs:return 0
            r=self.regs[cue]
            if mode=='simple':return r.answer()[int(weight<0)]
            if mode=='multiply':return r.compare(0,3)*(1 if weight>0 else -1)
            return r.compare(0,3,latest=weight<0)
    return Transfer
