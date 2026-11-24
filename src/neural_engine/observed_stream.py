"""Isolated batch lanes have independent episode and selector states, same archived cell solver."""
from itertools import combinations
import json,numpy as np
from . import replay as rp
from .portable_readout import PortableReadout
from .contact import Row
from . import writer
from .integrity import need
def fresh(pair,decoded,x,blobs,*,row_type):
    cp,mem,snap=decoded; before=pair.data;gate_ref=json.dumps(snap,sort_keys=True)
    mm=rp.FrozenMemory(mem);mm.keys.flags.writeable=False;mm.val.flags.writeable=False
    obj=rp.Filter(cp,'selective',mm);obj.reset(48);obj.setup(48,'all')
    obj.w.flags.writeable=False;obj.ring.flags.writeable=False
    n=obj.n;native=n.step;g=rp.Paired(snap,48,'pair')
    ix=np.flatnonzero(obj.w);need(len(ix)==1,'projection')
    lr=list(combinations(range(8),2))[int(ix[0])];w=int(obj.w[ix[0]])
    need(len(blobs)==48 and all(type(b) is writer.Bank for b in blobs),'per-stream payload')
    port=PortableReadout(obj,pair,'shared')
    port.rows=[row_type(r.lr,r.w,b) for r,b in zip(port.rows,blobs)]
    need(all(tuple(row.lr)==tuple(lr) and row.w==w for row in port.rows) and not hasattr(port,"episode_credit"),"wrong frozen port")
    sp=np.empty((48,288),np.uint16);cur=np.empty((48,288));en=np.empty((48,288),bool);ready=np.empty_like(en)
    stage=np.zeros((48,288),np.uint32);draft=np.zeros_like(stage);legacy=np.zeros_like(stage)
    cb=0
    def physical(row,current,learn):
        nonlocal cb
        need(learn is False and current.shape==(48,16) and row.shape==(48,8) and cb<288,'callback')
        allow=g.permits(row);en[:,cb]=allow;cur[:,cb]=current[:,8];ready[:,cb]=port.prepared!=0
        for i,r in enumerate(port.rows):
            r.latch.permission(allow[i])
            stage[i,cb]=r.latch.word;draft[i,cb]=r.live_offer
            m,f,l=r.credit;legacy[i,cb]=((m<<16)|((f+1)<<2)|(l+1)) if m else 0
        cb+=1
        return native(row,current*allow[:,None],False)
    n.step=physical
    with port.attached() as tick:
        for ti,row in enumerate(x.transpose(1,0,2)):
            count=n.count.copy();tick(row);bits=n.count-count
            need(np.all((bits==0)|(bits==1)),'binary events');sp[:,ti]=rp.compress16(bits)
    del n.step  # remove closure cycle; fresh solver for next batch
    need(cb==288 and n.payload()==cp['core'] and obj.payload()==cp['aux'] and mm.payload()==mem and
         not obj.writes and not n.dw_slots and not n.w.flags.writeable and not n.theta.flags.writeable and
         gate_ref==json.dumps(g.table,sort_keys=True) and pair.data==before,'frozen state or count changed')
    payout=[(r.latch.take(),r.latch.take()) for r in port.rows]
    return dict(sp=sp,cur=cur,en=en,ready=ready,stage=stage,draft=draft,legacy=legacy,payout=np.asarray(payout,dtype=np.uint32)),port
