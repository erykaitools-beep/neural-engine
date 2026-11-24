"""Read raw physical trials. Vectorised 3-sample gate oracle and independent scalar spike tally."""
from collections import Counter,defaultdict
from pathlib import Path
import json
import numpy as np
from .paths import ROOT
R=ROOT/'results/rhythm'
names=('clean','alt','block','late','burst','level','empty')
def need(x,s):
    if not x:raise ValueError(s)
def main(check):
    M=json.loads((R/'MATRICES.json').read_bytes())
    W=json.loads((ROOT/'configs/rhythm/weights.json').read_bytes())
    scores={};agg=defaultdict(Counter);paired=defaultdict(Counter);nsecond=0
    pc=np.array([i.bit_count() for i in range(256)],np.int16)
    with np.load(R/'INPUTS.npz',allow_pickle=False) as z:
        for sk,v in M.items():
            m=np.array(v['meta']);on=np.array(v['onsets']);slot=np.array(v['slots'])
            need(m.shape==(48,5) and len(set(map(tuple,m[:,:4])))==48,'histories')
            for carrier,bind in v['definitions'].items():
                check(True);x=z[f'inputs_{sk}_{carrier}'];extra=bind[3];phase=np.array(v['phase'][carrier])
                need(x.shape==(7,48,288,8) and np.isfinite(x).all() and (x>=0).all(),'input domain')
                mx=x.max(-1);strong=x>mx[...,None]/32
                weak_phys=np.where(strong,0.,x)
                need(np.all((weak_phys>0).sum(-1)<=1),'ambiguous residual channel')
                weak=weak_phys.max(-1)/np.where(mx>0,mx,1.);addr=strong@(1<<np.arange(8))
                for i,t in enumerate(on):
                    pulse=x[:,i,t:t+16];rh=pulse[1:5,:,extra]
                    need(np.all(x[:,i,:t]==x[0,i,:t]) and np.all(x[:,i,t+16:]==x[0,i,t+16:]),'unpaired history')
                    need(np.allclose(pulse[:6].sum(-1),pulse[0].sum(-1),rtol=0,atol=1e-12) and not pulse[6].any(),'dose or omission')
                    need(np.all(np.sort(pulse[1:5],axis=1)==np.sort(pulse[1:2],axis=1)) and np.all(rh>0),'not permutation of same per-channel samples')
                    hi=rh>rh.min(axis=1,keepdims=True);need(np.all(hi.sum(1)==8) and np.all(hi[:,0]==(phase[i]==0)),'amplitude or phase leak')
                    count=np.count_nonzero(hi[:,1:]!=hi[:,:-1],axis=1)
                    need(count.tolist()==[15,1,11,5],'bad class geometry')
                    for k,j,div in [(1,2,1),(1,4,5),(3,2,3)]:
                        need(np.flatnonzero((pulse[k]!=pulse[j]).any(-1))[0]==div,'distinctive tick')
                for mc in ('0','1'):
                    y=np.where(m[:,3]==int(mc),m[:,1],m[:,2])*2-1;ends={};clean=None
                    for mode in ('delay','memory','reverse-delay','reverse-memory'):
                        tag=f'{sk}_{carrier}_{mc}_{mode}'
                        if 'spikes_'+tag not in z:continue
                        rev=mode.startswith('reverse');ks=[1,2] if rev else list(range(7))
                        bits=z['spikes_'+tag];en=z['enable_'+tag];drive=z['drive_'+tag]
                        need(bits.shape==(len(ks),48,288) and bits.dtype==np.uint16,'spike shape')
                        if mode=='delay':clean=(bits[0].copy(),en[0].copy(),drive[0].copy())
                        if mode=='memory':need(all(np.array_equal(a,b) for a,b in zip(clean,(bits[0],en[0],drive[0]))),'clean unequal delay control')
                        if not rev:
                            for i,t in enumerate(on):
                                for k,j,div in [(1,2,1),(1,4,5),(3,2,3)]:
                                    for arr in (bits,en,drive):need(np.array_equal(arr[k,i,:t+div],arr[j,i,:t+div]),'noncausal prefix')
                        for j,k in enumerate(ks):
                            p=bits[j];addr_k=addr[k];wkey=('v' if carrier=='u' else 'u') if rev else carrier
                            ok=np.array(W[sk]['final'][wkey])[addr_k]>0
                            same=ok[:,:-1]&ok[:,1:]&(addr_k[:,:-1]==addr_k[:,1:])
                            if mode.endswith('memory'):
                                u,vv=weak[k,:,:-1],weak[k,:,1:];same &= ((u==0)&(vv==0))|((u>0)&(vv>0)&(abs(u-vv)>1/512.))
                            oracle=np.zeros_like(ok);oracle[:,2:]=same[:,:-1]&same[:,1:]
                            need(np.array_equal(oracle,en[j]) and np.array_equal(drive[j,:,:,1],drive[j,:,:,0]*oracle),'3-sample gate oracle/current')
                            forbidden=rev or k in (2,4,5,6);margin=(pc[p>>8]-pc[p&255]);score=[];cnt=Counter()
                            for i,t in enumerate(on):
                                w=margin[i,t:t+20]*y[i];nz=np.flatnonzero(w);sp=np.flatnonzero(p[i,t:t+20]);fs=int(sp[0]) if len(sp) else -1;fm=int(nz[0]) if len(nz) else -1
                                premature=bool(p[i,:t+2].any());silent=not p[i].any()
                                first=bool(len(nz) and w[fm]>0);vote=bool(w.sum()>0);hit=not premature and first and vote
                                yy=int(2*m[i,(1 if m[i,3]==int(mc) else 2)]-1)
                                ss=[int(a) for a in p[i,t:t+20]];ww=[yy*((a>>8).bit_count()-(a&255).bit_count()) for a in ss]
                                fm2=next((b for b,h in enumerate(ww) if h),-1)
                                pre2=any(p[i,:t+2].tobytes());sil2=not any(p[i].tobytes())
                                need((fs,fm,premature,silent,first,vote)==(next((b for b,h in enumerate(ss) if h),-1),fm2,pre2,sil2,fm2!=-1 and ww[fm2]>0,sum(ww)>0),'second raw scorer disagrees');nsecond+=1
                                correct=bool(silent) if forbidden else hit;score.append(correct)
                                row=Counter(n=1,correct=int(correct),silent=int(silent),premature=int(premature),false_alarm=int(forbidden and not silent),miss=int(not forbidden and not hit),delivered=int(drive[j,i,t:t+16,1].any()),opens=int(oracle[i,t:t+16].sum()),before_div5=int(fs>=0 and fs<5),conflict_n=int(m[i,1]!=m[i,2]),conflict_correct=int(m[i,1]!=m[i,2] and correct))
                                row['spike_offset_'+str(fs)]+=1;row['margin_offset_'+str(fm)]+=1
                                cnt.update(row)
                                for suffix in ('','/panel'+str(v['panel']),'/phase'+str(phase[i]),'/slot'+str(slot[i])):agg[mode+'/'+names[k]+suffix].update(row)
                            scores[tag+'/'+names[k]]=dict(cnt);ends[mode,k]=np.array(score)
                        check(True)
                    for k,name in enumerate(names):
                        a,b=ends['delay',k],ends['memory',k]
                        paired[name].update(n=len(a),gain=int((~a&b).sum()),loss=int((a&~b).sum()),both_ok=int((a&b).sum()),both_fail=int((~a&~b).sum()))
    need(len(scores)==256 and nsecond==256*48 and all(c['n']==768 for c in paired.values()),'score coverage')
    return dict(groups=scores,summary={k:dict(v) for k,v in agg.items()},paired={k:dict(v) for k,v in paired.items()},scalar_crosscheck_rows=nsecond)
