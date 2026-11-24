"""Offline regression tests; no learning or public services."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from neural_engine import circuit, spatial_protocol as spatial, replay, credit_port, writer
from neural_engine.association import Durable
from neural_engine.manifest import read_pins, verify
from neural_engine.integrity import need
from neural_engine.paths import ROOT as PACKAGE_ROOT

def read(name):return json.loads((ROOT/name).read_bytes())
class ReleaseTests(unittest.TestCase):
    def test_manifest_rejects_escape_and_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);p=path/"sample";p.write_bytes(b"ok")
            h=hashlib.sha256(b"ok").hexdigest();m=path/"SHA256SUMS"
            m.write_text(h+"  sample\n");verify(read_pins(path))
            p.write_bytes(b"changed")
            with self.assertRaises(ValueError):verify(read_pins(path))
            for invalid in ("../outside","/outside","sample\n"+h+"  sample"):
                m.write_text(h+"  "+invalid+"\n")
                with self.assertRaises(ValueError):read_pins(path)
    def test_spatial_generator_and_restore(self):
        panels=read("results/xyz/PANELS.json")
        self.assertEqual(ROOT, PACKAGE_ROOT)
        for sk,p in panels.items():
            d=spatial.dataset(int(sk));self.assertEqual(d["test"],p["test"])
            self.assertEqual(list(map(lambda e:spatial.tensor_sha(spatial.stimulus(e)),d["test"])),p["test_sha256"])
        sk="22401";p=read("results/xyz/cp_22401_xyz_64.json")
        class Clock:
            def tick(self):pass
        raw=spatial.readout(p,panels[sk]["test"][:4],Clock())
        self.assertEqual(raw["counts"],read("results/xyz/seed_22401.json")["rows"]["xyz"]["64"]["counts"][:4])
    def test_gate_threshold_edges(self):
        replay.fixtures_t()
    def test_rhythm_saved_batch(self):
        sk="202611203";state=read("configs/rhythm/start.json")[sk]
        self.assertEqual(circuit.clone(state["cp"]["core"]).payload(),state["cp"]["core"])
        weights=read("configs/rhythm/weights.json")[sk]["final"]["u"]
        with np.load(ROOT/"results/rhythm/INPUTS.npz",allow_pickle=False) as z:
            observed=replay.simulate(state["cp"],state["contexts"]["0"],replay.Temporal(weights,True,48),z[f"inputs_{sk}_u"][2])
            for kind,data in zip(("spikes","drive","enable"),observed):
                np.testing.assert_array_equal(data,z[f"{kind}_{sk}_u_0_memory"][2])
    def test_confirmation_before_feedback(self):
        root=ROOT/"results/confirmation";cfg=ROOT/"configs/confirmation"
        live=read("configs/rhythm/start.json")["202611203"]
        decoded=(live["cp"],live["contexts"]["0"],read("configs/confirmation/gate.json"))
        pair=(cfg/"pair.bin").read_bytes();trial=read("results/confirmation/PANEL.json")
        stage,k,lanes=trial["schedule"][0];self.assertEqual((stage,k),("test",0))
        held={}
        for m in (10,160):
            with np.load(root/f"held_{m}.npz",allow_pickle=False) as z:held[m]=z["x"]
        x=np.stack([held[m][i] for a,m,i in lanes]);self.assertEqual(len(x),48)
        banks=[writer.Bank((root/f"bank_0_{a}.bin").read_bytes(),diverse=True) for a,m,i in lanes]
        seen,_=credit_port.build(pair)(Durable(pair),decoded,x,banks)
        with np.load(root/"raw_00.npz",allow_pickle=False) as z:
            self.assertEqual(sorted(seen),sorted(z.files))
            for key,v in seen.items():np.testing.assert_array_equal(v,z[key])
    def test_confirmation_stimulus_generator(self):
        from neural_engine import stimuli
        cfg=ROOT/"configs/confirmation/source";model=read("configs/confirmation/source/PANELS.json")["202611203_0"]
        for m in (10,160):
            x,rows=stimuli.build(202611213,"random",m,model,(cfg/model["wave"]).read_bytes())
            self.assertEqual(rows,read("results/confirmation/PANEL.json")["rows"][str(m)])
            with np.load(ROOT/f"results/confirmation/held_{m}.npz",allow_pickle=False) as z:np.testing.assert_array_equal(x,z["x"])

if __name__=="__main__":unittest.main()
