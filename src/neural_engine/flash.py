"""Frozen N2 sensory motion probe. No classifier fitting, teaching or updates."""
import hashlib
import time

SEEDS=tuple(range(22541,22549))
ARMS=('xyz','blind','x')
KINDS=('motion','arrival','last_motion','last_arrival')
T,N,LAG=80,16,6

def need(x,msg):
    if not x: raise ValueError(msg)

def digest(x):
    return hashlib.sha256(x.tobytes(order='C')).hexdigest()

def cycle(order,np):
    g=np.empty((N,2),dtype=np.int64)
    for i,a in enumerate(order): g[a]=sorted([order[i-1],order[(i+1)%N]])
    return g

def layout(seed,np):
    pos=np.random.default_rng(seed+73000).random((N,3))
    def walk(k):
        route=[0]; remaining=set(range(1,N))
        while remaining:
            i=min(remaining,key=lambda j:(float(((pos[j,:k]-pos[route[-1],:k])**2).sum()),j))
            route.append(i);remaining.remove(i)
        return route
    gr={'xyz':cycle(walk(3),np),'x':cycle(walk(1),np),
        'blind':cycle([0,*np.random.default_rng(seed+93000).permutation(range(1,N))],np)}
    w=.05+.95*np.exp(-.5*((pos[:,0,None]-(np.arange(8)+.5)/8)/.18)**2)
    return pos,gr,w

def inputs(seed,np):
    rng=np.random.default_rng(seed+247000); rows=[]
    for k in ('motion','arrival'):
        for pair in range(12):
            a=float(rng.uniform(3.2,4.8));d=int(rng.integers(9,14));start=int(rng.integers(4,9))
            if k=='motion':
                left=int(rng.integers(5));fwd=list(range(left,left+4));rev=fwd[::-1]
            else:
                end=3+pair%2;fwd=list(range(end-3,end+1));rev=list(range(end+3,end-1,-1))
            for path,y in [(fwd,1),(rev,-1)]:
                for last in (False,True):
                    rows.append(dict(kind=('last_' if last else '')+k,pair=pair,y=y,
                        pixel=path,amp=a,dwell=d,onset=start))
    rows=[rows[i] for i in rng.permutation(len(rows))]
    v=np.zeros((len(rows),T,8),dtype='<f8');index={}
    for i,r in enumerate(rows):
        for j,b in enumerate(r['pixel']):
            if r['kind'].startswith('last_') and j<3: continue
            t=r['onset']+j*r['dwell'];v[i,t:t+r['dwell'],b]=r['amp']
        index[(r['kind'],r['pair'],r['y'])]=i
    for j in range(12):
        for k in KINDS:
            a,b=(index[(k,j,y)] for y in (1,-1));ra=rows[a];t=ra['onset']+3*ra['dwell']
            if k=='motion':need(np.array_equal(v[a].sum(0),v[b].sum(0)),'not dose matched')
            if k=='arrival':need(np.array_equal(v[a,t:],v[b,t:]),'terminal mismatch')
            if k=='last_arrival':need(np.array_equal(v[a],v[b]),'null inputs differ')
        for y in (1,-1):
            for k in ('motion','arrival'):
                i=index[k,j,y];h=index['last_'+k,j,y];t=rows[i]['onset']+3*rows[i]['dwell']
                need(not v[h,:t].any() and np.array_equal(v[i,t:],v[h,t:]),'terminal-only mismatch')
    need(not v[:,:4].any() and not v[:,-19:].any(),'common boundaries')
    return rows,v,index

def evidence(z,x,np):
    # For simultaneous cells at the two instants count every ordered pair.
    sign=np.sign(x[None,:]-x[:,None]).astype(np.int32)
    past=z[:,:-LAG].astype(np.int32);later=z[:,LAG:].astype(np.int32)
    return ((past@sign)*later).sum(axis=(1,2),dtype=np.int64)

def direction_metrics(rows,score,z):
    ans={}
    for k in KINDS:
        ix=[i for i,r in enumerate(rows) if r['kind']==k]
        hit=lambda i:int(score[i])*rows[i]['y']>0
        ans[k]=dict(n=len(ix),correct=sum(hit(i) for i in ix),
            right=sum(hit(i) for i in ix if rows[i]['y']==1),left=sum(hit(i) for i in ix if rows[i]['y']==-1),
            pairs_both=sum(all(hit(i) for i in ix if rows[i]['pair']==p) for p in range(12)),
            ties=sum(int(score[i])==0 for i in ix),silent=sum(not z[i].any() for i in ix),
            spikes=sum(int(z[i].sum()) for i in ix))
    return ans

def run(H,g,R,save):
    np=H.np;C=H.C
    # This run is small enough to seal the complete input evidence before any observation.
    panels={};cache={}
    for s in SEEDS:
        p,gr,w=layout(s,np);rows,v,idx=inputs(s,np);g.check()
        panels[str(s)]=dict(positions=p.tolist(),graphs={a:t.tolist() for a,t in gr.items()},
            W=w.tolist(),thresholds=[.4]*N,rows=rows,input_sha256=digest(v),
            sample_sha256=[digest(a) for a in v],input_dtype='<f8',shape=list(v.shape))
        cache[s]=(p,gr,w,rows,idx)
    save('PANELS.json',panels)
    fake=np.zeros((4,T,N),dtype=np.uint8)
    fake[0,1,0]=fake[0,1+LAG,15]=fake[1,1,15]=fake[1,1+LAG,0]=1
    fake[2,1,3]=fake[2,1+LAG,3]=1
    need(evidence(fake,np.arange(N),np).tolist()==[1,-1,0,0],'signed decoder fixture')
    details={};agg={a:{k:{} for k in KINDS} for a in ARMS}
    for si,s in enumerate(SEEDS):
        p,graphs,w,rows,index=cache[s]; rows2,v,_=inputs(s,np)
        need(rows2==rows and digest(v)==panels[str(s)]['input_sha256'],'generation mismatch')
        v.flags.writeable=False;sub={};zall={}
        for a in ARMS[si%3:]+ARMS[:si%3]:
            cpu=time.process_time(); gr=graphs[a]
            net=C.Network(N,s,'local','pull',gr);net.w=w.copy()
            need(np.all(net.theta==.4) and np.all((w>=.05)&(w<=1.)),'shared state')
            need(np.all(np.bincount(gr.ravel(),minlength=N)==2) and all(i in gr[j] for i in range(N) for j in gr[i]),'degree/reciprocity')
            route=net.route(np.eye(N,dtype=np.int64))
            need(all(set(np.flatnonzero(route[i]))==set(gr[i]) for i in range(N)),'physical impulse route')
            for prop in ('w','theta','neighbors'):getattr(net,prop).flags.writeable=False
            snapshot=net.payload()
            def evaluate(x):
                net.reset(len(x));zz=np.empty((len(x),T,N),dtype=np.uint8)
                for t in range(T):
                    zz[:,t,:]=net.step(x[:,t],teacher=None,learn=False);g.check()
                net.check();need(np.array_equal(zz.sum(1),net.count),'event accounting')
                return zz,dict(net.work)
            z,work=evaluate(v);rep,_=evaluate(v[:2])
            need(np.array_equal(rep,z[:2]) and snapshot==net.payload(),'reset/immutability')
            need(work['plastic_slots']==work['teacher_dose']==0 and work['cell_updates']==T*N*len(v),'no learning/teacher budget')
            scores=evidence(z,p[:,0],np)
            need(np.array_equal(scores,-evidence(z[:,::-1],p[:,0],np)),'time reversal of recorded spikes')
            for i in range(12):
                ai,bi=(index['last_arrival',i,y] for y in (1,-1))
                need(np.array_equal(z[ai],z[bi]),'null spike train mismatch')
            metrics=direction_metrics(rows,scores,z); masks=(z.astype(np.int32)@(1<<np.arange(N,dtype=np.int32))).tolist()
            sub[a]=dict(metrics=metrics,score=scores.tolist(),spike_mask_by_sample_tick=masks,
                work=work,graph_mean_length=float(np.linalg.norm(p[:,None]-p[gr],axis=2).mean()),
                cpu_s_including_checks=time.process_time()-cpu,reset_checked=2)
            for k in KINDS:
                for name,num in metrics[k].items(): agg[a][k][name]=int(agg[a][k].get(name,0)+num)
            zall[a]=z
        m=[i for i,r in enumerate(rows) if r['kind']=='motion']
        s0=tuple(np.sign(sub[a]['score']).astype(int) for a in ('xyz','blind'))
        delta=sub['xyz']['metrics']['motion']['correct']-sub['blind']['metrics']['motion']['correct']
        details[str(s)]=dict(delta_correct=delta,different_motion_traces=int(np.any(zall['xyz'][m]!=zall['blind'][m],axis=(1,2)).sum()),
            different_motion_decisions=int((s0[0][m]!=s0[1][m]).sum()))
        save('seed_'+str(s)+'.json',dict(seed=s,arms=sub,contrast=details[str(s)]));g.check()
    main=agg['xyz']['motion'];dn=main['n']
    diff=main['correct']-agg['blind']['motion']['correct']
    wins=sum(e['delta_correct']>0 for e in details.values())
    checks=dict(accuracy=main['correct']/dn>=.75,delta=diff/dn>=.05,wins=wins>=6,
        both_directions=all(agg['xyz'][k][dir]*3>=dn for k in ('motion','arrival') for dir in ('right','left')),
        beats_endpoint=(main['correct']-agg['xyz']['last_motion']['correct'])/dn>=.10,
        identical_null=all(agg[a]['last_arrival']['correct']<=dn/2 for a in ARMS))
    return dict(aggregate=agg,paired=details,delta_pp=100*diff/dn,wins=wins,criteria=checks,
        screen_for_replication=all(checks.values()),training=False,energy_J=None)
