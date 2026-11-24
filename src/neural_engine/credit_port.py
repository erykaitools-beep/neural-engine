"""No input from teacher, scorer or schedule. No write method for durable state."""
from functools import partial
from . import adapter,observed_stream,writer
from .integrity import need
P=1<<24
class Latch:
    # A single uint32 logical word; Python object overhead is not the packed payload.
    __slots__=('word',)
    def __init__(self):self.word=0
    def offer(self,token):
        if self.word<P:self.word=token
    def permission(self,actual):
        if actual and self.word<P:self.word=self.word+P if self.word else 3*P
    def take(self):
        if self.word//P!=1:return 0
        value=self.word%P;self.word=2*P;return value
def code(m,c,f,l):
    return m<<16|c<<8|(f+1)<<2|l+1
def controls():
    a=Latch();x=code(10,11,1,-1);y=code(160,138,-1,1)
    a.offer(x);need(not a.take(),'premature consumption');a.permission(False);need(not a.take(),'veto')
    a.offer(y);a.permission(True);a.offer(x);a.permission(True)
    need(a.take()==y and not a.take(),'one-shot binding')
    a.offer(x);a.permission(True);need(not a.take(),'double permission')
    b=Latch();b.permission(True);b.offer(x);b.permission(True);need(not b.take(),'missing at first authorization')
    return ['premature','veto','latest valid before release','freeze after release','double consumption','second permission','empty first release','fresh episode']
def build(pair):
    source=adapter.build('simple',pair)
    class Seen(source):
        def __init__(self,lr,w,blob):
            super().__init__(lr,w,bytes(4));need(type(blob) is writer.Bank,'lane bank view');self.bank=blob;self.latch=Latch();self.live_offer=0
        def read(self,memory,cue,mix):
            value=super().read(memory,cue,mix)
            packet=0
            if self.repeat_query and self.marker in self.bank and cue in self.regs:
                f,l=self.regs[cue].answer()
                if f and l:packet=code(self.marker,cue,f,l)
            self.latch.offer(packet)
            self.live_offer=packet  # comparator/instrument only, never consumed by the candidate
            return value
    return partial(observed_stream.fresh,row_type=Seen)
