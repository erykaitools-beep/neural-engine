"""Recalculate archived outcomes without running a training or simulation loop."""
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from neural_engine.manifest import verify, read_pins

STUDIES = ("xyz", "rhythm", "confirmation", "flash", "cohort", "prefix")


def run(name, pins):
    import numpy as np
    data = ROOT / "results" / name
    read = lambda f: json.loads((data / f).read_bytes())
    if name == "xyz":
        from neural_engine.score_xyz import run as score
        return score()
    if name == "flash":
        from neural_engine.score_flash import run as score
        score()
        return {"status": "matched"}
    if name == "rhythm":
        from neural_engine.score_rhythm import main
        v = main(lambda *_: None)
        if v != read("SCORES.json"): raise ValueError("rhythm score mismatch")
        return {"rows": v["scalar_crosscheck_rows"], "paired": v["paired"]}
    if name == "confirmation":
        from neural_engine.score_confirmation import recount
        v = recount(data, pins, None)
        if json.loads(json.dumps(v)) != read("OFFLINE_CHECK.json"):
            raise ValueError("confirmation score mismatch")
        return {k:v[k] for k in ("veto", "veto_emissions", "retention")}
    if name == "cohort":
        from neural_engine import cohort
        v = cohort.summarize(data)
        if v != read("summary.json"): raise ValueError("cohort summary mismatch")
        import gzip, hashlib
        original = json.loads((ROOT/"configs/cohort/uncompressed-sha256.json").read_text())
        for f,d in original.items():
            if hashlib.sha256(gzip.decompress((data/(f+".gz")).read_bytes())).hexdigest()!=d:
                raise ValueError("cohort original bytes mismatch")
        return {k:v[k] for k in ("acquisition_pass","retention_pass","joint_pass")}
    if name == "prefix":
        cfg = json.loads((ROOT/"configs/cell.json").read_bytes())
        scale = cfg["theta_p"] - cfg["theta_d"]
        out = {}
        for mode,n_equal in [("common0",59),("immediate24",237),("delayed24",264)]:
            large = equal = total = 0
            for s in range(11100,11103):
                with np.load(data/f"{s}_{mode}.npz",allow_pickle=False) as z:
                    i,j=z["pairs"].T
                    if sorted([*i,*j])!=list(range(240)) or not ((z["y"][i]==0)&(z["y"][j]==1)).all():
                        raise ValueError("pair identity")
                    ca=z["63_ca"];ras=z["raster"]
                    if ras.shape!=(240,161) or ca.shape!=(240,16,8):raise ValueError("shape")
                    large+=int((abs(ca[i]-ca[j]).max(axis=(1,2))/scale >= .01).sum())
                    equal+=int((ras[i,64:]==ras[j,64:]).all(axis=1).sum())
                    total+=len(i)
            if (total,large,equal)!=(360,360,n_equal):raise ValueError("prefix contrast")
            out[mode]={"pairs":total,"distinct_pre_Q_Ca_at_1pct":large,"identical_Q_to_end_raster":equal}
        return out
    raise ValueError(name)


if __name__ == "__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("study", nargs="?", choices=("all","integrity",*STUDIES), default="all")
    args=ap.parse_args()
    pins=read_pins(ROOT); verify(pins)
    print(f"SHA-256: {len(pins)} release files match", flush=True)
    for name in STUDIES if args.study=="all" else () if args.study=="integrity" else (args.study,):
        print(name, json.dumps(run(name,pins),sort_keys=True), flush=True)
