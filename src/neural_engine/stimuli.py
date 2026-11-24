"""Offline-only panel builder; never imported by adapter. Physical source checked against annotations."""
import itertools,random,io
import numpy as np
from .integrity import need,digest
def relation(s):
    a,b=s.find('00'),s.find('11');y,z=s.rfind('00'),s.rfind('11')
    return (0,0) if min(a,b)<0 else (1 if a<b else -1,1 if y>z else -1)
def signcode(t):return 2*int(t[0]>0)+int(t[1]>0)
def oo(seed):
    rng=random.Random(93182741+seed);n=21+2*(seed%2);words=set();pairs=[]
    signs=list(itertools.product((-1,1),repeat=2))
    for u,v in itertools.product(signs,repeat=2):
        pair=[]
        for target in (u,v):
            for _ in range(3000):
                s=''.join(rng.choice('01') for j in range(n))
                if s not in words and relation(s)==target:words.add(s);pair.append(s);break
            else:raise ValueError('finite stimulus cap')
        pairs.append(pair)
    plan=[]
    for i in range(n):
        addr=[0,1];rng.shuffle(addr);plan.extend((a,i) for a in addr)
    schedule=list(zip(plan,np.linspace(4,137,2*n).astype(int),[rng.uniform(.77,1.23) for _ in plan]))
    return pairs,schedule
def build(seed,kind,cue,src,npz):
    with np.load(io.BytesIO(npz),allow_pickle=False) as z:prefix=z['prefix'];tails=z['tail']
    alpha=src['alphabet'];base=[]
    if kind!='random':
        for i,q in enumerate(src['rows']):
            if q['phases']!=[signcode(q['targets'][1]),signcode(q['targets'][0])]:continue
            need(digest(prefix[i])==q['prefix'],'source prefix');base.append((prefix[i],q['words']))
    else:
        words,sched=oo(seed)
        for pair in words:
            x=np.zeros((141,8))
            for (a,i),t,amp in sched:
                mask=alpha[a][int(pair[a][i])];x[t,[j for j in range(8) if mask>>j&1]]=amp
            base.append((x,pair))
    base.sort(key=lambda p:tuple(relation(w) for w in p[1]))
    need(len(base)==16 and len({tuple(relation(w) for w in words) for _,words in base})==16,'joint crossed labels')
    indexes={(q['c'],q['go']):i for i,q in enumerate(src['borrowed'])};rows=[];waves=[]
    for i,(pref,words) in enumerate(base):
        for a,go in [(0,True),(1,True),((i%4+i//4)%2,False)]:
            x=np.concatenate((pref,tails[indexes[a,go]]),axis=0)
            need(not x[162:166].any() and not x[161].any() and not x[166].any(),'second marker slot occupied')
            x[162:166,[k for k in range(8) if cue>>k&1]]=.91
            hist=(pref>0)@(1<<np.arange(8))
            need([''.join(str(al.index(int(k))) for k in hist if k in al) for al in alpha]==words,'rendered physical word')
            tags=relation(words[a]);rows.append(dict(rid=digest(x),words=words,addr=a,go=go,f=tags[0],r=tags[1],marker=cue))
            waves.append(x)
    return np.stack(waves),rows
def window(x,spec,sign,addr):
    four=np.flatnonzero((x>0).sum(axis=1)==4)
    if not len(four):return None,None
    need(np.all(np.diff(four)==1),'split GO')
    high=x[four,spec['extra']]*80>x[four][:,spec['support']].max(1)
    ends=np.flatnonzero((high[2:]==high[1:-1])&(high[2:]==high[:-2]))+2
    if not len(ends) or bool(high[ends[0]])!=(sign*(1-2*addr)>0):return None,None
    rev=[j for j in ends if high[j]!=high[ends[0]]]
    return (int(four[rev[0]]),int(four[-1])+1) if rev else (None,None)
