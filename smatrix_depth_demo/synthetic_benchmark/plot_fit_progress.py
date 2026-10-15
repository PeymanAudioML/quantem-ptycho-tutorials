"""Live plot of a running joint S-matrix + probe fit, from its log (run_benchmark.py smatrix --verbose).

    python plot_fit_progress.py --log smatrix_P3_XY_full80.log --config configs/poly_si_3grain.yaml \
        --out results/smatrix_progress_P3_XY.png --every 60

Redraws the PNG every `--every` seconds while the log grows (stops when the run's summary line appears, or with
--once).  Panels: training loss per epoch (log scale; warm-up shaded) and the probe corrections dC = C - nominal for
C10, C12a, C12b of each non-reference dataset, with the true values (truth - nominal) as dashed lines.
"""
import argparse, re, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import bench

EPOCH = re.compile(r"epoch\s+(\d+)\s+loss\s+([0-9.eE+-]+)(?:\s+dC \[A\] (.*))?")


def parse(path):
    ep, loss, dc, stop, done = [], [], {}, None, None
    for line in open(path, errors="ignore"):
        m = EPOCH.search(line)
        if m:
            e = int(m.group(1))
            ep.append(e)
            loss.append(float(m.group(2)))
            if m.group(3):
                rows = re.findall(r"\[([^\[\]]+)\]", m.group(3))
                dc[e] = np.array([[float(v) for v in r.split()] for r in rows])
        if "early stop" in line:
            stop = line.strip()
        if re.match(r"\[smatrix\S*\] \d+ of", line):
            done = line.strip()
    return np.array(ep), np.array(loss), dc, stop, done


def draw(a, cfg):
    nominal, true, ref = bench.coefficients(cfg)
    names = ["C10", "C12a", "C12b"]
    tdc = (true - nominal)[:, :3]
    ep, loss, dc, stop, done = parse(a.log)
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.6))
    warm = a.warmup
    if len(ep):
        ax[0].semilogy(ep, loss, "k.-")
        ax[0].axvspan(0.5, warm + 0.5, color="0.9", label=f"warm-up (S only, {warm} epochs)")
        ax[0].set_xlim(0.5, max(ep.max(), warm + 2) + 0.5)
    ax[0].set_xlabel("epoch"), ax[0].set_ylabel("training loss (normalised)"), ax[0].legend(fontsize=8)
    ax[0].set_title(f"loss: epoch {ep.max() if len(ep) else 0}, {loss[-1]:.4g}" if len(ep) else "waiting for epoch 1")
    eps = sorted(dc)
    cols = plt.cm.tab10.colors
    for j, nm in enumerate(names):
        a_ = ax[1 + j]
        for d in range(tdc.shape[0]):
            if d == ref:
                continue
            if eps:
                a_.plot(eps, [dc[e][d, j] for e in eps], ".-", color=cols[d], label=f"dataset {d} fit")
            a_.axhline(tdc[d, j], color=cols[d], ls="--", lw=1.2, label=f"dataset {d} truth {tdc[d, j]:+g} Å")
        a_.axhline(0, color="0.6", lw=0.6)
        a_.set_xlabel("epoch"), a_.set_ylabel(f"d{nm} = fit − nominal (Å)"), a_.legend(fontsize=8)
        a_.set_title(f"{nm} correction (dataset {ref} = gauge, fixed)")
    status = done or stop or "running"
    fig.suptitle(f"{a.log}: {status[:160]}", fontsize=10)
    fig.tight_layout()
    fig.savefig(a.out, dpi=100)
    plt.close(fig)
    return bool(done)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--config", default=bench.DEFAULT_CONFIG)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=float, default=60.0)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    cfg = bench.load_config(a.config)
    while True:
        finished = draw(a, cfg)
        if a.once or finished:
            break
        time.sleep(a.every)


if __name__ == "__main__":
    main()
