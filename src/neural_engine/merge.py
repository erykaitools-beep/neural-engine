"""Post-GO local change. No episode number, arm name, task label, or external memory."""
from . import writer

def deliver(blob,ticket,target):
    updated=writer.deliver(blob,ticket,target,diverse=True,defer=True)
    if len(blob)!=5 or ticket==0 or target is None:
        return updated
    owner,x,p,child,old=writer.struct.unpack("=BbBBb",blob)
    y=writer.struct.unpack_from("=b",updated,4)[0]
    if ticket>>16==child and x*old>0 and x*y>0 and abs(y)>abs(old) and abs(y)>=2:
        return writer.pack(owner,x)
    return updated

def controls():
    # These are pure small transition fixtures, not extra presentations to the network.
    pack=writer.pack
    def token(m,first,last):return m<<16 | 11<<8 | (first+1)<<2 | (last+1)
    c=token(160,-1,+1);o=token(10,-1,+1)
    cases=[
      ('removes negative redundant child',pack(10,-1,0,160,-1),c,1,pack(10,-1)),
      ('positive owner mirror',pack(10,1,0,160,1),c,-1,pack(10,1)),
      ('temporary sign equality',pack(10,-1,0,160,0),c,1,pack(10,-1,0,160,-1)),
      ('opposing rule',pack(10,-1,0,160,1),c,-1,pack(10,-1,0,160,2)),
      ('wrong address',pack(10,-1,0,160,-2),o,1,pack(10,-2,0,160,-2)),
      ('equal votes',pack(10,-1,0,160,-2),token(160,1,1),1,pack(10,-1,0,160,-2)),
      ('weakening agreement',pack(10,-1,0,160,-3),c,-1,pack(10,-1,0,160,-2)),
      ('no payout',pack(10,-1,0,160,-2),0,1,pack(10,-1,0,160,-2)),
      ('no teacher',pack(10,-1,0,160,-2),c,None,pack(10,-1,0,160,-2)),
      ('silent owner',pack(10,0,0,160,-2),c,1,pack(10,0,0,160,-3))]
    for tag,b,t,y,want in cases:writer.need(deliver(b,t,y)==want,tag)
    return [v[0] for v in cases]
