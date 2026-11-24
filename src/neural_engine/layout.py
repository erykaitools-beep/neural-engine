"""Preregistered external curriculum, never supplied to the neuronal reader."""
ARMS=('noise_single','noise_double','shift_single','shift_double')
ROWS=(15,19,21,24)
def selected(qs):
    seen=set();out=[]
    for i,q in enumerate(qs):
        group='veto' if not q['go'] else 'conflict' if q['f']!=q['r'] else 'agree'
        key=(group,q['addr'],q['f'])
        if group=='conflict' or key not in seen:out.append(i)
        seen.add(key)
    if len(out)!=24:raise ValueError('strata')
    return out
def batches(rows):
    inds=selected(rows)
    def evaluate(k,a):return ('test',k,[(a,m,i) for m in (10,160) for i in inds])
    out=[evaluate(0,a) for a in ARMS]
    for k,i in enumerate(ROWS,1):
        out.append(('learn',k,[(a,160,i) for a in ARMS]))
        out.extend(evaluate(k,a) for a in ARMS)
    return out
