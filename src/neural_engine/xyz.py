"""Paired geometry test. Static wiring only. Calls the authenticated N2 training and readout, never its old runner."""
import hashlib,json,time
from collections import Counter
from itertools import combinations

ARMS=('xyz','xy','blind')
SEEDS=(22401,22402,22403,22404)
POINTS=(0,16,32,48,64)

def require(x,s):
    if not x:raise ValueError(s)

def make_cycle(order,np):
    require(len(order)==16 and set(order)==set(range(16)),'permutation')
    a=np.empty((16,2),dtype=np.int64)
    for j,i in enumerate(order):a[i]=sorted((order[j-1],order[(j+1)%16]))
    return a

def nearest(points,d):
    order=[0];remaining=set(range(1,16))
    while remaining:
        # Minimize squared Euclidean distance; ID tie break. Never see W, labels or outputs.
        a=points[order[-1],:d]
        i=min(remaining,key=lambda j:(float(((points[j,:d]-a)**2).sum()),j))
        order.append(i);remaining.remove(i)
    return order

def graphs(seed,np):
    coords=np.random.default_rng(seed+73000).random((16,3))
    orders={a:nearest(coords,len(a)) for a in ('xyz','xy')}
    orders['blind']=[0,*map(int,np.random.default_rng(seed+93000).permutation(range(1,16)))]
    return coords,{a:make_cycle(orders[a],np) for a in ARMS}

def averages(counts,tests,np):
    c=np.asarray(counts)
    require(c.shape==(32,16) and (c>=0).all() and (c<=80).all(),'count shape/range')
    scores=c.reshape(32,2,8).sum(-1);tied=scores[:,0]==scores[:,1]
    pred=np.where(tied,-1,np.argmax(scores,axis=1))
    both={}
    for phase in ('A','B'):
        ix=[i for i,row in enumerate(tests) if row['phase']==phase]
        labels=[tests[i]['label'] for i in ix]
        require(len(ix)==16 and sum(labels)==8,'heldout balance')
        both[phase]=dict(correct=int((pred[ix]==labels).sum()),ties=int(tied[ix].sum()),
          silent=int((c[ix].sum(axis=1)==0).sum()),spikes=int(c[ix].sum()),n=16)
    return both

def run(H,g,R,save):
    np=H.np;C=H.C
    class Pace:
        def tick(self):g.check()
    pace=Pace()
    saved=lambda name:json.loads((R/name).read_bytes())
    sha=lambda name:hashlib.sha256((R/name).read_bytes()).hexdigest()
    # Freeze all panels before measuring any arm.
    panels={}
    for s in SEEDS:
        d=H.dataset(s);used=Counter();train=[]
        for e in d['train']:
            k=e['phase'],e['label']
            if used[k]<16:train.append(e);used[k]+=1
        test=d['test'];xx=lambda e:H.tensor_sha(H.stimulus(e))
        tr,te=[list(map(xx,rr)) for rr in (train,test)]
        require(len(train)==64 and len(test)==32 and [e['phase'] for e in train]==['A']*32+['B']*32 and
                len(set(tr))==64 and len(set(te))==32 and not set(tr)&set(te),'split integrity')
        pos,gr=graphs(s,np)
        panels[str(s)]=dict(train=train,test=test,train_sha256=tr,test_sha256=te,
          xyz=pos.tolist(),graphs={a:v.tolist() for a,v in gr.items()})
    save('PANELS.json',panels)
    rows={};costs={a:0. for a in ARMS}
    for si,s in enumerate(SEEDS):
        p=panels[str(s)];pos=np.asarray(p['xyz']);nets={};length={};route={};raw={a:{} for a in ARMS}
        journal={a:[] for a in ARMS};cpfiles={}
        for a in ARMS:
            graph=np.asarray(p['graphs'][a]);net=C.Network(16,s,policy='local',routing='pull',graph=graph)
            C.graph_check(graph,16)
            require(np.array_equal(np.bincount(graph.ravel(),minlength=16),np.full(16,2)) and
                    all(i in graph[j] for i in range(16) for j in graph[i]),'degree/reciprocity')
            # Deliver one isolated spike from each possible source; verify actual routing, not just neighbor list.
            impulse=net.route(np.eye(16,dtype=np.int64));expected=np.zeros((16,16),dtype=np.int64)
            for target,src in enumerate(graph):expected[src,target]=1
            require(np.array_equal(impulse,expected),'one-spike routing')
            route[a]=impulse.tolist();net.reset();nets[a]=net
            length[a]=float(np.linalg.norm(pos[:,None,:]-pos[graph],axis=2).mean())
        require(all(np.array_equal(nets[a].w,nets['xyz'].w) and np.array_equal(nets[a].theta,nets['xyz'].theta) for a in ARMS),'initial W/theta')
        def capture(k):
            for a,n in nets.items():
                f=f'cp_{s}_{a}_{k}.json';save(f,n.payload());cpfiles[f]=sha(f)
                cp=saved(f);r=H.readout(cp,p['test'],pace)
                H.validate_work(r['work'],16,batch=32);v=averages(r['counts'],p['test'],np)
                require(cp['neighbors']==p['graphs'][a] and r['payload_sha256']==C.digest(n.payload()),'restart identity')
                raw[a][str(k)]=dict(counts=r['counts'],strict=v,legacy=r['tasks'],work=r['work'],checkpoint=f)
        capture(0)
        for j,e in enumerate(p['train'],1):
            shift=(j+si)%3
            for a in ARMS[shift:]+ARMS[:shift]:
                t=time.process_time();_,v=H.train(nets[a],e,pace);costs[a]+=time.process_time()-t
                H.validate_journal(v,16)
                journal[a].append(dict(dW=v['W_delta'],dtheta=v['theta_delta'],preview=v['preview_counts']))
            if j in POINTS:capture(j)
        # Per-seed independent sources available to offline reviewers.
        name=f'seed_{s}.json'
        save(name,dict(seed=s,checkpoint_hashes=cpfiles,one_spike_routing=route,mean_length3D=length,rows=raw,journal=journal))
        rows[str(s)]={a:dict(initial={t:raw[a]['0']['strict'][t]['correct'] for t in 'AB'},
          A32=raw[a]['32']['strict']['A']['correct'],Afinal=raw[a]['64']['strict']['A']['correct'],
          Bfinal=raw[a]['64']['strict']['B']['correct'],ties=sum(raw[a]['64']['strict'][t]['ties'] for t in 'AB'),
          silent=sum(raw[a]['64']['strict'][t]['silent'] for t in 'AB'),length3D=length[a]) for a in ARMS}
        g.check(True)
    diffs={};stats={}
    for a in ARMS:
        vs=[r[a] for r in rows.values()]
        stats[a]=dict(A32=sum(v['A32'] for v in vs),Afinal=sum(v['Afinal'] for v in vs),Bfinal=sum(v['Bfinal'] for v in vs),
          final_per_task_n=64,delta_A_count=sum(v['A32']-v['Afinal'] for v in vs),both_pass=sum(v['Afinal']>=13 and v['Bfinal']>=13 for v in vs),
          acquired_A32=sum(v['A32']>=13 for v in vs),lost_A=sum(v['A32']>=13 and v['Afinal']<13 for v in vs),
          mean_length3D=sum(v['length3D'] for v in vs)/4,training_cpu_s=costs[a])
    for a in ('xy','blind'):
        ds=[((r['xyz']['Afinal']+r['xyz']['Bfinal'])-(r[a]['Afinal']+r[a]['Bfinal']))/32 for r in rows.values()]
        diffs[a]=dict(paired=ds,mean=sum(ds)/4,positive=sum(x>0 for x in ds))
    screen=all(v['mean']>=.05 and v['positive']>=3 and stats['xyz']['delta_A_count']<=stats[a]['delta_A_count'] for a,v in diffs.items())
    return dict(per_seed=rows,summary=stats,contrasts=diffs,preregistered_benefit_screen=screen,n_neurons=16,inhibitory_arcs=32,
       checkpoints_per_arm=len(POINTS),dimensions_setup_bytes=dict(xyz=384,xy=256,blind=0),geometry_runtime_distance_calls=0,energy_J=None)
