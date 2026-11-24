"""Internal worker for reproduce.py; no service code or archived run reservations."""
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from neural_engine.integrity import need
from neural_engine.manifest import read_pins,verify
from neural_engine import spatial_protocol as spatial

class Budget:
    steps=0
    cell_updates=0
    def check(self,*_):
        if time.process_time()>84: raise TimeoutError("worker CPU budget")

def reference(opts,out):
    study=opts['study'];g=Budget();r=ROOT/'results'/study
    get=lambda p:json.loads(p.read_bytes())
    def save(n,v):
        with (out/n).open('x') as f:f.write(json.dumps(v,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n')
    if study in ('xyz','flash'):
        from neural_engine import xyz,flash
        v=(xyz if study=='xyz' else flash).run(spatial,g,out,save)
        save('RESULT.json',{'summary':v})
        from neural_engine import score_xyz,score_flash
        verifier=score_xyz if study=='xyz' else score_flash
        if study=='xyz':verifier.ROOT=out
        else:verifier.R=out
        verifier.run()
        need(get(out/'PANELS.json')==get(r/'PANELS.json'),'generated inputs')
        # Compare every scientific output, excluding only measured host-dependent CPU seconds.
        def scientific(v):
            if isinstance(v,dict):return {k:scientific(x) for k,x in v.items() if k not in ('training_cpu_s','cpu_s_including_checks')}
            if isinstance(v,list):return list(map(scientific,v))
            return v
        for p in r.glob('*.json'):
            need(scientific(get(out/p.name))==scientific(get(p)),'fresh mismatch: '+p.name)
    elif study=='rhythm':
        from neural_engine import replay as rp
        sk=str(opts['seed']);s=get(ROOT/'configs/rhythm/start.json')[sk];ws=get(ROOT/'configs/rhythm/weights.json')[sk]['final']
        checked=0
        with np.load(r/'INPUTS.npz',allow_pickle=False) as z:
            for carr in ('u','v'):
                panels=z[f'inputs_{sk}_{carr}']
                for mc in ('0','1'):
                    for mode in ('delay','memory','reverse-delay','reverse-memory'):
                        tag=f'{sk}_{carr}_{mc}_{mode}'
                        if 'spikes_'+tag not in z:continue
                        weights=ws[('u' if carr=='v' else 'v') if mode.startswith('reverse') else carr]
                        ks=(1,2) if mode.startswith('reverse') else range(7)
                        expected=[z[k+'_'+tag] for k in ('spikes','drive','enable')]
                        for j,k in enumerate(ks):
                            gate=rp.Temporal(weights,mode.endswith('memory'),48)
                            seen=rp.simulate(s['cp'],s['contexts'][mc],gate,panels[k])
                            need(all(np.array_equal(a,b[j]) for a,b in zip(seen,expected)),'rhythm mismatch: '+tag)
                            checked+=1;g.check()
        need(checked==64,'missing batches');save('parity.json',{'seed':int(sk),'matched_batches':checked})
    elif study=='confirmation':
        from neural_engine import confirm_protocol as cp
        result=cp.run(g,out)
        for f in sorted(r.iterdir()):
            if f.suffix=='.npz':
                with np.load(f,allow_pickle=False) as a,np.load(out/f.name,allow_pickle=False) as b:
                    need(set(a.files)==set(b.files) and all(np.array_equal(a[k],b[k]) for k in a.files),'fresh array mismatch: '+f.name)
            elif f.suffix=='.bin':need(f.read_bytes()==(out/f.name).read_bytes(),'bank mismatch')
            elif f.name in ('TEACHING.json','RELOAD.json','PANEL.json'):need(get(f)==get(out/f.name),'journal mismatch')
        save('parity.json',result)
    elif study=='prefix':
        from neural_engine import cohort as c
        s=opts['seed'];data=c.panel(s);_,x,y=c.tensors(data['test'])
        cohort=c.load(ROOT/'results/cohort'/f'result_{s}.json')
        for mode,policy,point in [('common0','immediate','0'),('immediate24','immediate','24'),('delayed24','delayed','24')]:
            cp=cohort['checkpoints']['local/'+policy][point]
            with np.load(r/f'{s}_{mode}.npz',allow_pickle=False) as z:
                left,right=z['pairs'].T
                need(np.array_equal(y,z['y']) and np.array_equal(x[left,44:],x[right,44:]),'matched suffix')
                for cut in (None,44,64):
                    net=c.N.clone(cp);net.reset(len(x))
                    for f in ('w','theta','neighbors'):getattr(net,f).flags.writeable=False
                    raster=np.zeros((len(x),161),dtype=np.uint16)
                    for t in range(161):
                        if t==cut:net.reset(len(x))
                        prev=net.count.copy();net.step(x[:,t],None,False)
                        raster[:,t]=(net.count-prev).astype(np.uint16)@(1<<np.arange(16,dtype=np.uint16))
                        if t==63 and cut is None:need(np.array_equal(net.ca,z['63_ca']),'pre-Q calcium')
                        g.check()
                    if cut is None:need(np.array_equal(raster,z['raster']),'raster')
                    else:need(np.array_equal(raster[left,cut:],raster[right,cut:]),'history survived reset')
                    need(net.payload()==cp,'frozen state changed')
        save('parity.json',{'seed':s,'states':3,'saved_raster_and_calcium_match':True})
    else:
        from neural_engine import cohort as c
        s,arm,pol=opts['seed'],opts['arm'],opts['policy'];d=c.panel(s)
        _,xt,yt,_,xe,ye,xo=c.data_check(d); old=np.tile([0,1,0,1],10)
        key=arm+'/'+pol;net=c.N.clone(c.checkpoint(s,arm));saved=c.load(r/f'result_{s}.json')
        def check(k):
            p=net.payload();need(p==saved['checkpoints'][key][str(k)],'checkpoint mismatch')
            need(c.evaluate(p,xt,xe,xo,yt,ye,old)==saved['rows'][key][str(k)],'readout mismatch')
            save(f'cp_{k}.json',p)
        check(0)
        for k,i in enumerate(d['order'],1):
            g.check()
            lab=d['teacher_labels'][i] if pol=='label_shuffle' else int(yt[i])
            start=128 if pol=='delayed' else d['time_starts'][i] if pol=='time_shuffle' else 64
            journal=c.teach(net,xt[i],lab,start,pol=='frozen')
            need(journal==saved['journals'][key][k-1],'lesson mismatch')
            if k in c.POINTS:check(k)
        save('parity.json',{'seed':s,'condition':key,'matched_lessons':96})

if __name__=='__main__':
    # A direct invocation must not accidentally start an unsupervised fresh experiment.
    import os, resource
    if not hasattr(os,'sched_getaffinity') or len(os.sched_getaffinity(0))!=1 or resource.getrlimit(resource.RLIMIT_CPU)[1]!=85 or resource.getrlimit(resource.RLIMIT_AS)[1]!=1<<30:
        raise SystemExit("Use experiments/reproduce.py and the site safety checklist.")
    verify(read_pins(ROOT))
    output=Path(sys.argv[2]).resolve()
    if output.is_relative_to(ROOT) or not output.is_dir() or any(output.iterdir()):raise ValueError("expected new separate output")
    reference(json.loads(sys.argv[1]),output)
