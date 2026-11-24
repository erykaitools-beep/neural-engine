"""Read-only inference candidate: available separately; no network, filesystem or clocks here.
Shares archived onset and downstream callback with the native reference in readout_port.
mode is a fixed experimental arm, never an episode label. Refuse ambiguous collapsed keys.
"""
from itertools import combinations
import numpy as np
from .association import Durable,Episode,P,need
from .readout_port import Readout
class ProjectionEpisode(Episode):
    def __init__(self,lr,w,mode):
        super().__init__(lr,w);self.mode=mode
    def read(self,memory,cue,mix):
        if self.mode=='original':return super().read(memory,cue,mix)
        if self.mode=='first_only':mix=(1.,0.)
        if self.mode=='last_only':mix=(0.,1.)
        # Never choose an address from an identity oracle. Every nonempty stored slot
        # is scanned; zero/negative entries count for capacity but contribute no drive.
        z=0
        for (_,a,b),orientation in self.evidence(cue,mix):
            for source,i,j,weight in P.iter_unpack(memory.data):
                if source and i==a and j==b:z+=max(0,weight)*orientation
        return (z>0)-(z<0)

class PortableReadout:
    # Pin the original qualified implementation of tick/attachment, not a fork.
    attached=Readout.attached
    def __init__(self,choice,weights,mode):
        need(type(weights) is Durable and mode in ('original','shared','first_only','last_only'),'frozen contact/control')
        physical=list(P.iter_unpack(weights.data))
        if mode!='original':
            ids=[(i,j) for c,i,j,w in physical if c]
            need(len(ids)==len(set(ids)),'ambiguous pair across cue addresses')
        ww=choice.w;ix=np.flatnonzero(ww)
        need(ww.shape==(28,) and len(ix)==1 and ww[ix[0]] in (-1,1) and choice.norm==1,'physical projection')
        lr=list(combinations(range(8),2))[int(ix[0])];w=int(ww[ix[0]])
        need(choice.last.ndim==2 and choice.last.shape[1]==8 and not choice.last.any(),'fresh episode required')
        self.c=choice;self.weights=weights
        self.rows=[ProjectionEpisode(lr,w,mode) for _ in choice.last]
        self.prepared=np.zeros(len(self.rows),np.int8)
