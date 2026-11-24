"""Task-independent first-discovery and recency order; one canonical uint16 per alphabet."""
from math import perm
class Invalid(ValueError): pass
OFF=(0,1,5,17,41)
BASE=(0,1,2,3)
def rank(seq,pool):
    avail=list(pool);n=len(seq);r=0
    for i,v in enumerate(seq):
        j=avail.index(v);r+=j*perm(len(avail)-1,n-i-1);avail.pop(j)
    return r
def unrank(r,n,pool):
    avail=list(pool);out=[]
    if not 0<=r<perm(len(avail),n):raise Invalid('rank')
    for k in range(n-1,-1,-1):
        j,r=divmod(r,perm(len(avail)-1,k));out.append(avail.pop(j))
    return tuple(out)
def pack(f,r,p,a):
    return OFF[len(f)]+rank(f,BASE) | rank(r,sorted(f))<<7 | p<<12 | int(a)<<14
def unpack(v):
    fr=v&127;rr=(v>>7)&31;p=(v>>12)&3;a=bool(v&16384)
    if v>>15 or fr>=65 or p==3 or (p==0 and (a or fr)):raise Invalid('fields')
    n=next(i for i in (4,3,2,1,0) if OFF[i]<=fr)
    f=unrank(fr-OFF[n],n,BASE);r=unrank(rr,n,sorted(f))
    if r and (r[-1]%2+1!=p):raise Invalid('last destination')
    return f,r,p,a
class Register:
    __slots__=('a','b','v')
    def __init__(self,a,b,blob=b'\x00\x00'):
        if not (0<a<256 and 0<b<256 and a!=b) or type(blob) is not bytes or len(blob)!=2:raise Invalid('constructor')
        self.a,self.b=a,b;self.v=int.from_bytes(blob,'little');unpack(self.v)
    def tick(self,mask):
        q=1 if mask==self.a else 2 if mask==self.b else 0
        if not q:self.v&=~16384;return
        f,r,p,active=unpack(self.v)
        if active and q==p:return
        if p:
            e=2*(p-1)+q-1
            if e not in f:f+=e,
            r=tuple(i for i in r if i!=e)+(e,)
        self.v=pack(f,r,q,True)
    def compare(self,left,right,latest=False):
        if left not in BASE or right not in BASE or left==right:raise Invalid('query')
        order=unpack(self.v)[int(latest)]
        if left not in order or right not in order:return 0
        s=1 if order.index(left)<order.index(right) else -1
        return -s if latest else s
    def snapshot(self):return self.v.to_bytes(2,'little')
