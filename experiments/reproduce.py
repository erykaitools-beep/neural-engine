"""Run one bounded reference experiment on a laboratory host under independent monitoring."""
import argparse, os, signal, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def host_ok(sensor, unit, before):
    raw=sensor.read_text().strip()
    if not raw.isdecimal():raise ValueError("unreadable temperature")
    t=int(raw)/1000
    if not (0<t<(60 if before else 75)):raise RuntimeError(f"thermal gate: {t} C")
    if unit:
        p=subprocess.run(["systemctl","show",unit,"-p","LoadState","-p","ActiveState"],text=True,capture_output=True,timeout=1)
        if p.returncode or set(p.stdout.splitlines())!={"LoadState=loaded","ActiveState=inactive"}:
            raise RuntimeError("specified service not verifiably idle")
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("study",choices=["xyz","flash","rhythm","confirmation","cohort","prefix"])
    ap.add_argument("--output",type=Path,required=True,help="new directory OUTSIDE the checkout")
    ap.add_argument("--sensor",type=Path,required=True,help="Linux CPU hwmon input (millidegrees C)")
    site=ap.add_mutually_exclusive_group(required=True)
    site.add_argument("--idle-unit",help="systemd workload unit that MUST stay inactive")
    site.add_argument("--dedicated-host",action="store_true",help="only on a separate host with no protected workload")
    ap.add_argument("--seed",type=int)
    ap.add_argument("--arm",choices=["baseline","local"],default="local")
    ap.add_argument("--policy",choices=["immediate","delayed","frozen","label_shuffle","time_shuffle"],default="immediate")
    ap.add_argument("--allow-learning",action="store_true",help="explicit opt-in for weight changes in a new output only")
    a=ap.parse_args()
    if sys.platform!="linux":ap.error("fresh-run supervisor requires Linux; archived scoring is portable")
    import resource
    if a.study in ("xyz","confirmation","cohort") and not a.allow_learning:ap.error("this protocol requires --allow-learning")
    if a.study=="cohort" and a.seed not in range(11100,11130):ap.error("cohort requires one archived --seed per command")
    if a.study=="prefix" and a.seed not in (11100,11101,11102):ap.error("prefix requires one of its three archived seeds")
    if a.study=="rhythm" and a.seed not in (202611202,202611203,202611262,202611263):ap.error("rhythm requires one archived --seed per command")
    if a.study in ("xyz","flash","confirmation") and a.seed is not None:ap.error("this protocol has a fixed set of seeds")
    s=a.sensor.resolve(strict=True)
    if not s.is_relative_to("/sys/devices") or not s.name.startswith("temp") or not s.name.endswith("_input"):
        ap.error("use an actual CPU sensor in sysfs; confirm its identity for your machine")
    out=a.output.resolve()
    if out.is_relative_to(ROOT) or not out.parent.is_dir():ap.error("choose a new directory outside the checkout")
    host_ok(s,a.idle_unit,True)
    out.mkdir(mode=0o700,exist_ok=False)
    env=dict(os.environ,OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",PYTHONDONTWRITEBYTECODE="1")
    # Bound the child (not a system-wide energy or hard real-time guarantee).
    import ctypes
    libc=ctypes.CDLL(None,use_errno=True);parent=os.getpid()
    def limits():
        if libc.prctl(1,signal.SIGKILL,0,0,0)!=0 or os.getppid()!=parent:os._exit(120)
        os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});os.nice(10)
        for res,lim in [(resource.RLIMIT_AS,1<<30),(resource.RLIMIT_CPU,85),(resource.RLIMIT_FSIZE,120<<20),(resource.RLIMIT_CORE,0)]:
            resource.setrlimit(res,(lim,lim))
    import json
    opts=json.dumps(dict(study=a.study,seed=a.seed,arm=a.arm,policy=a.policy))
    proc=None;start=time.monotonic()
    try:
        proc=subprocess.Popen([sys.executable,"-B",str(ROOT/"experiments/_reference.py"),opts,str(out)],env=env,
                              start_new_session=True,preexec_fn=limits)
        def check_budget():
            host_ok(s,a.idle_unit,False)
            if time.monotonic()-start>=300:raise TimeoutError("wall time limit")
            if sum(p.stat().st_size for p in out.iterdir() if p.is_file())>=120<<20:raise RuntimeError("output limit")
        while proc.poll() is None:
            time.sleep(.05)
            if proc.poll() is not None:break
            os.killpg(proc.pid,signal.SIGSTOP)
            check_budget()
            time.sleep(.20)
            check_budget()
            if proc.poll() is None:os.killpg(proc.pid,signal.SIGCONT)
        check_budget()
        if proc.returncode:raise SystemExit(proc.returncode)
        print("MATCH:",out)
    finally:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid,signal.SIGKILL);proc.wait()
if __name__=="__main__":main()
