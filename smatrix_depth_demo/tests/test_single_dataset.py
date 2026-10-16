import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "synthetic_benchmark"))

import run_benchmark as benchmark
import run_realization as realization


def config():
    return dict(name="test", specimen=dict(thickness_A=200.0),
                defocus=dict(nominal_C10=[-166.7, -100.0, -33.3], reference=0),
                _path="test.yaml", _sha256="test")


def test_load_only_selected_dataset(monkeypatch):
    loaded = []
    setup = SimpleNamespace(K=4, dr=0.2)
    positions = np.array([[[0.4, 0.4], [0.4, 0.6]]])

    def load(path, args, dataset):
        loaded.append((path, dataset))
        assert dataset == 1
        return dict(data=np.ones((1, 2, 4, 4)), dk=[1.25, 1.25], zero=[2, 2],
                    positions_A=positions, defocus=100.0)

    monkeypatch.setattr(benchmark.bench, "make_setup", lambda cfg: setup)
    monkeypatch.setattr(benchmark.bench, "specimen_dir", lambda cfg, name: "/unused")
    monkeypatch.setattr(realization, "load_dataset", load)
    patterns, positions_px, offset, focus = benchmark.load_variant(config(), "sample", "window", "noiseless", [1])
    assert loaded == [("/unused/window_noiseless/def1", 1)]
    assert patterns.shape == (1, 1, 2, 4, 4)
    np.testing.assert_allclose(focus, [100.0])
    np.testing.assert_allclose(positions_px * setup.dr, positions)
    np.testing.assert_array_equal(offset, [0, 0])


@pytest.mark.parametrize("extra", [[], ["--single-dataset"]])
def test_cli_defaults(monkeypatch, extra):
    calls = []
    monkeypatch.setattr(sys, "argv", ["run_benchmark.py", "quantem", *extra])
    monkeypatch.setattr(benchmark.bench, "load_config", lambda path: config())
    monkeypatch.setattr(benchmark, "quantem", lambda cfg, args: calls.append(args))
    benchmark.main()
    args = calls[0]
    assert args.dataset == 1
    assert args.arms == (["A"] if extra else list("ABCD"))
    assert args.probes == (["nominal_trained"] if extra else ["smatrix_trained"])


@pytest.mark.parametrize("extra", [
    ["--probes", "smatrix_trained"], ["--probes", "true_fixed"], ["--arms", "D"],
    ["--dataset", "3"], ["--dataset", "-1"], ["--num-slices", "0"],
])
def test_single_dataset_rejects_priors_and_invalid_options(monkeypatch, extra):
    monkeypatch.setattr(sys, "argv", ["run_benchmark.py", "quantem", "--single-dataset", *extra])
    monkeypatch.setattr(benchmark.bench, "load_config", lambda path: config())
    with pytest.raises(SystemExit) as error:
        benchmark.main()
    assert error.value.code == 2


def test_standalone_quantem_never_loads_smatrix(monkeypatch, tmp_path):
    import forward_check
    import quantem_export

    loaded = []
    coefficients = []
    updates = []
    setup = SimpleNamespace(K=4, dr=0.2, lam=0.02)
    dataset = dict(defocus=100.0, positions_A=np.array([[[1.0, 1.0]]]))

    def load(path, args, index):
        loaded.append(index)
        assert path.endswith("/def1")
        return dataset

    def probe(current_setup, nominal):
        coefficients.append(nominal.copy())
        return np.ones((4, 4), np.complex64)

    def make_pt(data, metadata, obj, initial_probe, device):
        return SimpleNamespace(
            obj_shape_full=obj.shape, sampling=np.array([0.2, 0.2]),
            dset=SimpleNamespace(scan_positions_px=torch.zeros((1, 2))),
            probe=initial_probe[None], obj_model=SimpleNamespace(obj=torch.as_tensor(obj)),
            iter_losses=[2.0, 1.0], reconstruct=lambda **kwargs: updates.append(kwargs))

    monkeypatch.setattr(benchmark.bench, "HERE", str(tmp_path))
    monkeypatch.setattr(benchmark.bench, "make_setup", lambda cfg: setup)
    monkeypatch.setattr(benchmark.bench, "specimen_dir", lambda cfg, name: str(tmp_path / "data"))
    monkeypatch.setattr(benchmark.bench.ad, "entrance_probe", probe)
    monkeypatch.setattr(benchmark.bench, "load_config", lambda path: config())
    monkeypatch.setattr(realization, "load_dataset", load)
    monkeypatch.setattr(forward_check, "setup_dataset", lambda data, lam: data)
    monkeypatch.setattr(forward_check, "make_pt", make_pt)
    monkeypatch.setattr(forward_check, "truth_probe_on_grid", lambda data, shape, dr, ref: data[0])
    monkeypatch.setattr(quantem_export, "to_quantem", lambda potential, grid: potential)
    monkeypatch.setattr(sys, "argv", ["run_benchmark.py", "quantem", "--single-dataset", "--device", "cpu",
                                     "--num-slices", "25", "--iters", "2", "--probe-warmup", "1"])
    benchmark.main()
    assert loaded == [1]
    assert len(coefficients) == 1 and coefficients[0][0] == -100.0
    assert "probe" not in updates[0]["optimizer_params"]
    assert "probe" in updates[1]["optimizer_params"]
    summaries = list((tmp_path / "results").rglob("summary.json"))
    assert len(summaries) == 1
    summary = json.loads(summaries[0].read_text())
    assert summary["smatrix_run"] is None
    assert summary["single_dataset"] and summary["dataset"] == 1
    assert summary["planes"] == 25 and summary["n_candidates"] == 0
    assert not summary["oracle"] and not summary["informed_potential"]
    assert summaries[0].parent.name.endswith("_single")
    np.testing.assert_allclose(np.diff(summary["slice_boundaries"]), 8.0)


def test_single_dataset_smatrix_keeps_geometry_guard():
    setup = benchmark.sj.Setup(energy_ev=300000, alpha=0.02, taper=0.002, K=96, Y=200, dr=0.2)
    axis = np.arange(48, 153, 2)
    positions = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1)
    nominal = np.zeros((1, len(benchmark.sj.NAMES)))
    nominal[0, 0] = -100.0
    with pytest.raises(ValueError, match="measurements/unknowns"):
        benchmark.sj.check_geometry(setup, positions, nominal, 200.0, abtem=True, strict=True, log=lambda text: None)
