"""Two-event weakening gate. No world/labels/timestep/row metadata. A PostAction is single-use upstream."""
from . import writer,merge
EMPTY=bytes(2)
def apply(bank,pending,ticket,teacher,verdict):
    if type(pending)!=bytes or len(pending)!=2 or not (pending==EMPTY or
            writer.key(pending[0]) and pending[1] in (1,255)):
        raise ValueError('invalid pending')
    if verdict is False or not ticket:return bank,EMPTY,'reject'
    marker=ticket>>16; first=((ticket>>2)&3)-1; last=(ticket&3)-1
    proposal=teacher*((first>last)-(first<last))
    view=writer.Bank(bank,diverse=True)
    if view.get(marker)*proposal<0:
        if verdict is not True:return bank,EMPTY,'no-bit-for-weakening'
        evidence=bytes((marker,proposal%256))
        if pending!=evidence:return bank,evidence,'defer'
        return merge.deliver(bank,ticket,teacher),EMPTY,'confirmed'
    return merge.deliver(bank,ticket,teacher),EMPTY,'ordinary'
def fixtures():
    b=writer.pack(10,-1,0,160,1)
    def t(m,f):return m<<16 | 11<<8 | (f+1)<<2 | (1-f)
    a,p,n=apply(b,EMPTY,t(160,-1),1,True)
    assert a==b and p==bytes((160,255)) and n=='defer'
    a2,p2,n=apply(a,p,t(160,1),-1,True)
    assert a2==writer.pack(10,-1,0,160,0) and p2==EMPTY and n=='confirmed'
    for v in (False,None):
        assert apply(a,p,t(160,1),-1,v)[:2]==(a,EMPTY)
    assert apply(a,p,t(160,1),1,True)[:2]==(writer.pack(10,-1,0,160,2),EMPTY)
    assert apply(a,p,t(10,1),1,True)[:2]==(a,bytes((10,1)))
    assert apply(a2,EMPTY,t(160,-1),1,None)[:2]==(writer.pack(10,-1,0,160,-1),EMPTY)
    return 'two distinct events with opposite raw targets; cancellation/replacement; silent zero'
