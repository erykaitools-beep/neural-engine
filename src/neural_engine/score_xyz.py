"""Recompute geometry, signed outcomes and scores from archived records; no engine or training imports."""
import hashlib
import itertools
import json
import math
from pathlib import Path

from .paths import ROOT
ROOT=ROOT/'results/xyz'

def read(name):
    return json.loads((ROOT / name).read_bytes())

def pinned(name, sha):
    if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != sha:
        raise ValueError(name)

def check(c, what):
    if not c:
        raise ValueError(what)

def run():
    panels, result = read('PANELS.json'), read('RESULT.json')['summary']
    arms = ('xyz', 'xy', 'blind')
    scores = {a:[] for a in arms}
    differences = {b:dict(different_spike_vectors=0,different_decisions=0,different_final_weights=0) for b in arms[1:]}
    all_train, all_test = set(), set()
    for s, panel in panels.items():
        seed = read('seed_'+s+'.json')
        for name, sha in seed['checkpoint_hashes'].items():
            pinned(name, sha)
        all_train.update(panel['train_sha256'])
        all_test.update(panel['test_sha256'])
        check(len(set(panel['train_sha256']))==64 and len(set(panel['test_sha256']))==32, 'split')
        positions = panel['xyz']
        earned, decisions = {}, {}
        for a in arms:
            graph=panel['graphs'][a]
            for i,row in enumerate(graph):
                check(len(set(row))==2 and i not in row and all(i in graph[j] for j in row),'degree')
            seen, frontier=set(),[0]
            while frontier:
                i=frontier.pop()
                if i not in seen:
                    seen.add(i)
                    frontier.extend(graph[i])
            check(len(seen)==16, 'not one cycle')
            if a != 'blind':
                walk=[0]
                while len(walk)<16:
                    walk.append(min(set(range(16))-set(walk),key=lambda j:(math.dist(positions[j][:len(a)],positions[walk[-1]][:len(a)]),j)))
                check(all(sorted(graph[i])==sorted([walk[k-1],walk[(k+1)%16]]) for k,i in enumerate(walk)), 'geometry')
            length=sum(math.dist(positions[i],positions[j]) for i,ns in enumerate(graph) for j in ns)/32
            check(abs(seed['mean_length3D'][a]-length)<1e-12,'length')
            route=[[int(i in r) for r in graph] for i in range(16)]
            check(route==seed['one_spike_routing'][a], 'one-spike edges')
            for k in ('0','16','32','48','64'):
                obs = seed['rows'][a][k]
                cp=read(obs['checkpoint'])
                check(cp['neighbors']==graph, 'checkpoint edges')
                counts=obs['counts']
                pred=[(0 if sum(v[:8])>sum(v[8:]) else 1 if sum(v[:8])<sum(v[8:]) else -1) for v in counts]
                decisions[a,k]=pred
                for t in ('A','B'):
                    ix=[i for i,e in enumerate(panel['test']) if e['phase']==t]
                    actual=dict(correct=sum(pred[i]==panel['test'][i]['label'] for i in ix), n=len(ix),
                        ties=sum(pred[i]==-1 for i in ix),silent=sum(sum(counts[i])==0 for i in ix),spikes=sum(sum(counts[i]) for i in ix))
                    check(actual==obs['strict'][t], 'strict outcomes')
                    earned[a,k,t]=actual['correct']
            scores[a].append([earned[a,'32','A'],earned[a,'64','A'],earned[a,'64','B']])
        for b,d in differences.items():
            left,right=(seed['rows'][a]['64'] for a in ('xyz',b))
            d['different_spike_vectors']+=sum(x!=y for x,y in zip(left['counts'],right['counts']))
            d['different_decisions']+=sum(x!=y for x,y in zip(decisions['xyz','64'],decisions[b,'64']))
            d['different_final_weights']+=read(left['checkpoint'])['weights']!=read(right['checkpoint'])['weights']
    check(len(all_train)==256 and len(all_test)==128 and not all_train&all_test, 'cross-seed split')
    gates=[]
    for a,s in scores.items():
        q=result['summary'][a]
        check([sum(v) for v in zip(*s)]==[q[k] for k in ('A32','Afinal','Bfinal')], 'totals')
        if a=='xyz':
            continue
        delta=[(sum(x[1:])-sum(y[1:]))/32 for x,y in zip(scores['xyz'],s)]
        check(delta==result['contrasts'][a]['paired'], 'paired contrast')
        gates.append(sum(delta)/4>=.05 and sum(v>0 for v in delta)>=3 and
                     sum(v[0]-v[1] for v in scores['xyz'])<=sum(v[0]-v[1] for v in s))
    check(all(gates)==result['preregistered_benefit_screen'],'decision')
    return dict(status='VERIFIED_SAVED_ONLY',checkpoints=60,panels=4,final_spike_vectors_per_arm=128,
        differences=differences,scores=scores,screen=all(gates))

if __name__ == '__main__':
    print(json.dumps(run(),sort_keys=True))
