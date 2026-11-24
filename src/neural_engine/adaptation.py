"""Single spike-triggered somatic feedback; not full AdEx or human calibration.
The pinned N1.0 core is unchanged. OLD a is subtracted from current this step;
then exact decay + spike jump. All sixteen a variables are transient, not weights.
"""
from pathlib import Path
import numpy as np
from .base_cell import Network as Baseline, config, config_id, relax

ROOT=Path(__file__).resolve().parent


class Network(Baseline):
    def __init__(self,seed,c=None,arm='full',adaptation=False):
        self.adaptation=bool(adaptation)
        super().__init__(seed,c,arm)
        if not np.isfinite(self.c['tau_adapt']) or self.c['tau_adapt']<=0 or not np.isfinite(self.c['adapt_jump']) or self.c['adapt_jump']<0:
            raise ValueError('invalid adaptation configuration')

    def reset(self,batch=1):
        super().reset(batch)
        self.a=np.zeros_like(self.s)
        self.teacher_spikes=0
        self.teacher_dose=0.
        self.adaptation_peak=0.

    def step(self,x,teacher=None,learn=False,nmda=True,vdcc=True):
        supplied=np.zeros_like(self.s) if teacher is None else np.asarray(teacher)
        old_counts=self.count.copy()
        used=super().step(x,supplied-self.a,learn,nmda,vdcc)
        spikes=self.count-old_counts
        jump=self.c['adapt_jump'] if self.adaptation else 0.
        self.a=self.a*np.exp(-self.c['dt']/self.c['tau_adapt'])+jump*spikes
        self.teacher_spikes+=int(spikes[supplied>0].sum())
        self.teacher_dose+=float(supplied.sum()*self.c['dt'])
        self.adaptation_peak=max(self.adaptation_peak,float(self.a.max()))
        if not np.isfinite(self.a).all():raise FloatingPointError('nonfinite adaptation')
        return used

    def episode(self,*args,**kwargs):
        pred,scores,diag,trace=super().episode(*args,**kwargs)
        diag.update(teacher_spikes=self.teacher_spikes,teacher_dose=self.teacher_dose,adaptation_peak=self.adaptation_peak)
        return pred,scores,diag,trace
