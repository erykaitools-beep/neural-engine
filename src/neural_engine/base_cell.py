"""N1.0 engineering model; arbitrary units, NOT a calibrated human neuron.

Order per step: exact input trace -> Euler coupled dendrite/soma (old voltages)
-> threshold/reset and backprop trace -> local Ca -> efficacy relaxation.
The calcium source gates use the NEW dendrite voltage. NMDA-like = conjunction
of presynaptic trace and voltage gate; VDCC-like = voltage gate only. Neither is
a conductance-channel/molecular model. Somatic spikes feed dendrites next step.

dw/dt = r(c)*p*(F(c)-w); step holds c,p fixed and uses exact exponential.
No SGD/backpropagation-of-error, target lookup or task ID inside this network.
The external teacher supplies class-selective current, a substantial assumption.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

from .paths import ROOT



def config():
    return json.loads((ROOT / 'configs/cell.json').read_text())


def config_id(c):
    return hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()


def relax(w, ca, pre, c):
    potentiating = ca >= c['theta_p']
    active = ca >= c['theta_d']
    rate = np.where(potentiating, c['rate_p'], np.where(active, c['rate_d'], 0.0))
    target = np.where(potentiating, c['w_high'], c['w_low'])
    alpha = -np.expm1(-c['dt'] * rate * pre)
    return w + alpha * (target - w)


class Network:
    def __init__(self, seed, c=None, arm='full'):
        self.c = dict(c or config())
        self.arm = arm
        if (self.c['cells'],self.c['inputs'],self.c['branches']) != (16,8,2):
            raise ValueError('unsupported dimensions')
        if arm not in ('full','frozen','zero_ca','shuffled_ca','unpaired','constant_ca'):
            raise ValueError('unknown arm')
        if not 0 < self.c['dt'] <= min(self.c['tau_d']/(1+self.c['coupling_d']),self.c['tau_s']/(1+self.c['coupling_s'])):
            raise ValueError('dt outside explicit voltage stability bound')
        self.w = np.random.default_rng(seed + 100000).uniform(
            self.c['init_low'], self.c['init_high'], (16, 8))
        self.rng = np.random.default_rng(seed + 200000)
        self.reset()

    def reset(self, batch=1):
        self.p = np.zeros((batch, 8))
        self.d = np.zeros((batch, 16, 2))
        self.s = np.zeros((batch, 16))
        self.bp = np.zeros_like(self.s)
        self.ca = np.zeros((batch, 16, 8))
        self.count = np.zeros_like(self.s)

    def step(self, x, teacher=None, learn=False, nmda=True, vdcc=True):
        c = self.c
        dt = c['dt']
        x = np.asarray(x).reshape(self.p.shape)
        if teacher is None:
            teacher = np.zeros_like(self.s)
        self.p = self.p * np.exp(-dt / c['tau_pre']) + x * (-np.expm1(-dt / c['tau_pre']))
        current = (self.p[:, None, :] * self.w[None, :, :]).reshape(-1, 16, 2, 4).sum(-1)
        old_d, old_s = self.d, self.s
        self.d = old_d + dt / c['tau_d'] * (
            -old_d + c['coupling_d'] * (old_s[:, :, None] - old_d) + current + self.bp[:, :, None])
        next_s = old_s + dt / c['tau_s'] * (
            -old_s + c['coupling_s'] * (old_d.mean(-1) - old_s) + teacher)
        spike = next_s >= c['spike_threshold']
        self.s = np.where(spike, 0.0, next_s)
        self.bp = self.bp * np.exp(-dt / c['tau_bp']) + c['bp_jump'] * spike
        self.count += spike
        voltage = np.repeat(self.d, 4, axis=-1)
        n_gate = np.clip((voltage - c['nmda_offset']) / c['nmda_scale'], 0, 1)
        v_gate = np.clip((voltage - c['vdcc_offset']) / c['vdcc_scale'], 0, 1)
        source = int(nmda) * c['nmda_gain'] * self.p[:, None, :] * n_gate + int(vdcc) * c['vdcc_gain'] * v_gate
        decay = np.exp(-dt / c['tau_ca'])
        self.ca = self.ca * decay + source * c['tau_ca'] * (1 - decay)
        used = self.ca[0]
        if self.arm == 'zero_ca':
            used = np.zeros_like(used)
        elif self.arm == 'constant_ca':
            used = np.ones_like(used)
        elif self.arm == 'shuffled_ca' and learn:
            used = self.rng.permutation(used.ravel()).reshape(16, 8)
        if learn and self.arm != 'frozen':
            if self.p.shape[0] != 1:
                raise ValueError('training is sequential, batch must equal one')
            self.w = relax(self.w, used, self.p[0], c)
        return used

    def episode(self, stimulus, label=None, learn=False, trace=False, nmda=True, vdcc=True, induction=False):
        """Input shape time,batch,8. All transient state is discarded per episode."""
        stimulus = np.asarray(stimulus)
        if label is not None and not learn and not induction:
            raise ValueError('teacher forbidden during evaluation; use explicit induction fixture')
        if stimulus.ndim == 2:
            stimulus = stimulus[:, None, :]
        self.reset(stimulus.shape[1])
        c = self.c
        start, end = (c['unpaired_start'], c['unpaired_end']) if self.arm == 'unpaired' else (c['teacher_start'], c['teacher_end'])
        records = []
        diag = dict(ca_max=0., used_changed=0, ltp_slots=0, ltd_slots=0, teacher_overlap=0., steps=0)
        for k, x in enumerate(stimulus):
            teacher = np.zeros_like(self.s)
            if label is not None and start <= k * c['dt'] < end:
                teacher[:, int(label)*8:(int(label)+1)*8] = c['teacher_amplitude']
            used = self.step(x, teacher, learn, nmda, vdcc)
            if learn or trace:
                diag['ca_max'] = max(diag['ca_max'], float(self.ca.max()))
                diag['used_changed'] += int(np.count_nonzero(used != self.ca[0]))
                diag['ltp_slots'] += int(np.count_nonzero((used >= c['theta_p']) & (self.p[0] > 1e-8)))
                diag['ltd_slots'] += int(np.count_nonzero((used >= c['theta_d']) & (used < c['theta_p']) & (self.p[0] > 1e-8)))
                diag['teacher_overlap'] += float(self.p.sum() * np.count_nonzero(teacher) * c['dt'])
                diag['steps'] += 1
            if trace:
                records.append(dict(t=k*c['dt'], d=self.d.tolist(), s=self.s.tolist(),
                                    bp=self.bp.tolist(), p=self.p.tolist(), ca=self.ca.tolist(),
                                    w=self.w.tolist(), counts=self.count.tolist()))
        scores = self.count.reshape(-1, 2, 8).sum(-1)
        if not all(np.isfinite(a).all() for a in (self.w, self.d, self.s, self.ca, self.bp)):
            raise FloatingPointError('nonfinite network state')
        if self.w.min() < c['w_low']-1e-12 or self.w.max() > c['w_high']+1e-12:
            raise FloatingPointError('efficacy out of bounds')
        return scores.argmax(-1), scores, diag, records

    def payload(self):
        return dict(kind='N1', config_id=config_id(self.c), weights=self.w.tolist())

    def restore(self, payload):
        if set(payload) != {'kind', 'config_id', 'weights'} or payload['kind'] != 'N1' or payload['config_id'] != config_id(self.c):
            raise ValueError('invalid persistent-only checkpoint')
        weights = np.asarray(payload['weights'], dtype=float)
        if weights.shape != (16, 8) or not np.isfinite(weights).all():
            raise ValueError('invalid weights')
        if weights.min() < self.c['w_low'] or weights.max() > self.c['w_high']:
            raise ValueError('weights outside domain')
        self.w = weights.copy()
        self.reset()
