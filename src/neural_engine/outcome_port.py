"""Post-action adapter; no scorer, bank, world, solver, or test inputs. Not in the live engine."""
class PostAction:
    __slots__ = ('eid','action','used')
    def __init__(self, eid, action):
        if type(eid) is not tuple or not eid or not (action is None or type(action) is int and action in (-1,1)):
            raise ValueError('bad committed event')
        self.eid,self.action,self.used=eid,action,False
    def consume(self, eid, teacher, success):
        if self.used or eid != self.eid: raise ValueError('reused or wrong event')
        if type(teacher) is not int or teacher not in (-1,1): raise ValueError('teacher')
        if not (success is None or type(success) is bool): raise ValueError('not an outcome bit')
        if self.action is None and success is not None: raise ValueError('no action for this outcome')
        self.used=True
        if success is None:return None
        return (self.action==teacher)==success

def fixtures():
    n=0
    for a in (-1,1):
        for t in (-1,1):
            for b in (True,False):
                assert PostAction((4,),a).consume((4,),t,b) is (t==(a if b else -a));n+=1
    for a in (-1,1,None):
        assert PostAction((4,),a).consume((4,),1,None) is None;n+=1
    for label in ('wrong','double','silent','invalid'):
        p=PostAction((4,),None if label=='silent' else 1)
        if label=='double':p.consume((4,),1,True)
        try:p.consume((3,) if label=='wrong' else (4,),1,1 if label=='invalid' else True)
        except ValueError as e:
            assert str(e)=={'silent':'no action for this outcome','invalid':'not an outcome bit'}.get(label,'reused or wrong event');n+=1
        else:raise AssertionError(label)
    return n
