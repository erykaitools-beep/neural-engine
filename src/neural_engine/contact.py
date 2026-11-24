"""Physical marker, two bounded addressed synapses, no access to targets during inference."""
from struct import Struct
from .portable_readout import ProjectionEpisode, PortableReadout
from .association import need
slot=Struct('Bb')
def table(blob):
    need(type(blob) is bytes and len(blob)==4,'selector payload')
    rows=list(slot.iter_unpack(blob))
    need(all((k.bit_count()==2 and -31<=w<=31) or (k==w==0) for k,w in rows),'invalid selector')
    vals={k:w for k,w in rows if k};need(len(vals)==sum(bool(k) for k,w in rows),'duplicate contacts')
    return vals
def update(blob,credit,direction):
    key,first,last=credit
    need(key.bit_count()==2 and first in (-1,0,1) and last in (-1,0,1) and direction in (-1,1),'teacher format')
    # Only outcome direction, not which candidate supplied it. Equal votes carry no differential credit.
    inc=direction*((first>last)-(first<last))
    if not inc:return blob
    t=table(blob);t[key]=max(-31,min(31,t.get(key,0)+inc))
    need(len(t)<=2,'bounded operator memory')
    return b''.join(slot.pack(k,v) for k,v in sorted(t.items())).ljust(4,b'\x00')
class Row(ProjectionEpisode):
    def __init__(self,lr,w,blob):
        super().__init__(lr,w,'shared');self.bank=table(blob);self.marker=0;self.credit=(0,0,0)
    def observe(self,k):
        if k.bit_count()==2:self.marker=k
        super().observe(k)
    def read(self,memory,cue,mix):
        if not self.marker:return super().read(memory,cue,mix)
        first=super().read(memory,self.marker,(1.,0.));last=super().read(memory,self.marker,(0.,1.))
        self.credit=(self.marker,first,last)
        weight=self.bank.get(self.marker,0)
        return first if weight>0 else last if weight<0 else 0
class Port(PortableReadout):
    def __init__(self,choice,pair,blob):
        super().__init__(choice,pair,'shared')
        self.rows=[Row(r.lr,r.w,blob) for r in self.rows]
