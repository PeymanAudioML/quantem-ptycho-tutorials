"""Paper-faithful tcBF / acBF (single dataset each) vs S-matrix recovery + refocusing, all structures."""
import os, json
import numpy as np
import paper_acbf as pa
import sweep_compare as sc

HERE = pa.HERE; OUT = os.path.join(HERE, "results_paper"); os.makedirs(OUT, exist_ok=True)
hand = json.load(open(os.path.join(OUT, "handedness.json")))["hand"]
CFG = [("baseline_3layers", "results"), ("six_phi1p5", "results_six_phi1p5"), ("six_phi2p5", "results_six_phi2p5"), ("poly", "results_poly")]
allres = {}
for label, rdir in CFG:
    dps, defoci, depths, score, smat = pa.setup("poly" if label == "poly" else label, os.path.join(HERE, rdir))
    stacks = {}
    for idf, df in enumerate(defoci):
        pr = pa.prep(dps[idf], hand)
        for kind in ("tcbf", "acbf"):
            stacks[f"{kind} paper (df={df:.0f})"] = pa.sections(pr, -df, depths, kind)
    stacks["acbf paper (3 df summed, extension)"] = sum(stacks[f"acbf paper (df={df:.0f})"] for df in defoci)
    stacks["S-matrix + refocusing"] = smat
    res = {k: score(v) for k, v in stacks.items()}
    allres[label] = res
    np.savez_compressed(os.path.join(OUT, f"stacks_{label}.npz"), depths=depths, **{k.replace(" ", "_"): v for k, v in stacks.items()})
    json.dump(allres, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    print("=====", label, flush=True)
    for k, v in res.items():
        if label == "poly":
            print(f"  {k:38s} depth err {v['mean_depth_err_A']:5.1f} A  mean r {v['mean_diag_r']:+.2f}  within10A {v['frac_within_10A']*100:3.0f}%  selectivity {v['selectivity']:+.2f}  inverted {v['n_inverted']}/{v['n_depths']}", flush=True)
        else:
            print(f"  {k:38s} resolved {v['n_resolved']}/{v['n_layers']}  depth err {v['mean_abs_depth_err']:5.1f} A (max {v['max_abs_depth_err']:5.1f})  mean|r| {v['mean_abs_r']:.2f}  inverted {v['n_contrast_inverted']}", flush=True)
