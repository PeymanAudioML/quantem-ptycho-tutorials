"""QuantEM with depth TV (QuantEM's built-in `tv_weight_z`), reusing run_benchmark.py unchanged.

For each weight w the shared runner is called as `run_benchmark.py quantem <args> --tag <tag>_tvz<w>`, with QuantEM's
reconstruct() patched so its object constraints are {"positivity": True, "tv_weight_z": w} instead of positivity only.
QuantEM adds the TV term to the loss it reports, so the TV part is recorded per batch and stored next to the run:
summary.json gains tv_weight_z, tv_loss_history, data_loss_history (= reported - TV) and final_data_loss.

TV term (quantem object_models.py _calc_tv_loss): w * mean_{k,y,x} |V[k+1,y,x] - V[k,y,x]| over the whole object array.

Run from synthetic_benchmark/, e.g.
  ../../.venv/bin/python regularized_experiment/run_quantem_tv.py --tv-z 0,10,100,1000 -- \
      --config configs/poly_si_3grain.yaml --specimen P3_XYZ --dataset 1 --smatrix-run <S run> --single-focus-handoff \
      --arms F --probes smatrix_trained --num-slices 20 --iters 200 --probe-warmup 40 --qlr 5e-3 --probe-lr 1e-4 \
      --qbatch 64 --seeds 0 --tag sfallQ20 --device cuda
"""
import argparse, glob, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
sys.path.insert(0, BENCH)
os.chdir(BENCH)

import run_benchmark  # noqa: E402
from quantem.diffractive_imaging.ptychography import Ptychography  # noqa: E402

STATE = dict(w=0.0, batch=[], iters=[])
_orig_reconstruct = Ptychography.reconstruct
_orig_soft = Ptychography._soft_constraints
_orig_record = Ptychography._record_iter


def _reconstruct(self, *args, constraints=None, **kw):
    cons = {k: dict(v) for k, v in (constraints or {}).items()}
    cons.setdefault("object", {})["tv_weight_z"] = STATE["w"]
    return _orig_reconstruct(self, *args, constraints=cons, **kw)


def _soft(self):
    val = _orig_soft(self)
    STATE["batch"].append(float(val.detach()))
    return val


def _record(self, iter_loss):
    STATE["iters"].append(float(np.mean(STATE["batch"])) if STATE["batch"] else 0.0)
    STATE["batch"] = []
    return _orig_record(self, iter_loss)


Ptychography.reconstruct = _reconstruct
Ptychography._soft_constraints = _soft
Ptychography._record_iter = _record


def wtag(w):
    return f"tvz{w:g}"


def main():
    if "--" not in sys.argv:
        raise SystemExit("usage: run_quantem_tv.py --tv-z w1,w2,... -- <run_benchmark quantem arguments incl. --tag>")
    k = sys.argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--tv-z", required=True, help="comma list of tv_weight_z values")
    a = ap.parse_args(sys.argv[1:k])
    rest = sys.argv[k + 1:]
    if "--tag" not in rest:
        raise SystemExit("pass --tag for the runner")
    ti = rest.index("--tag") + 1
    base_tag = rest[ti]
    for w in [float(v) for v in a.tv_z.split(",")]:
        STATE.update(w=w, batch=[], iters=[])
        args = list(rest)
        args[ti] = f"{base_tag}_{wtag(w)}"
        sys.argv = ["run_benchmark.py", "quantem"] + args
        print(f"=== tv_weight_z = {w:g}: run_benchmark.py quantem {' '.join(args)}", flush=True)
        run_benchmark.main()
        spec = args[args.index("--specimen") + 1]
        for f in glob.glob(os.path.join(BENCH, "runs", "*", spec, f"quantem_*_{args[ti]}*", "summary.json")) + \
                glob.glob(os.path.join(BENCH, "results", "quantem", "*", spec, f"quantem_*_{args[ti]}*", "summary.json")):
            s = json.load(open(f))
            rep = np.asarray(s["loss_history"], float)
            tv = np.asarray(STATE["iters"][-len(rep):], float)
            s.update(tv_weight_z=w, tv_loss_history=tv.tolist(), data_loss_history=(rep - tv).tolist(),
                     final_data_loss=float(rep[-1] - tv[-1]), final_tv_loss=float(tv[-1]),
                     regularization=f"QuantEM tv_weight_z={w:g} (object), positivity; via regularized_experiment/run_quantem_tv.py")
            json.dump(s, open(f, "w"), indent=1)
        print(f"=== tv_weight_z = {w:g} done: reported {rep[-1]:.4g}, TV {tv[-1]:.4g}, data {rep[-1] - tv[-1]:.4g}",
              flush=True)


if __name__ == "__main__":
    main()
