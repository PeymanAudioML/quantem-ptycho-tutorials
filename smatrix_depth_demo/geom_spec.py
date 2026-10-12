"""Acquisition/sampling setups for the abTEM polycrystal data (numpy only, importable from both the project env and the abTEM env).
a20: 20 mrad, 0.2 A pixels, 64 px / 12.8 A window, 25 x 25 scan at 0.8 A (the original setup; S grid 192^2, 529 beams).
a30: 30 mrad, 0.16 A pixels, 80 px / 12.8 A window, 25 x 25 scan at 0.8 A (S grid 240^2, ~1200 beams); abTEM band limit ~41 mrad.
Sample names: poly_abtem[D][_setup]  (D = number of defoci, default 3; setup default a20), e.g. poly_abtem7_a30."""
L_BOX = 38.4
SETUPS = {"a20": dict(alpha=0.020, dr=0.2, K=64, nscan=25, step=4),
          "a30": dict(alpha=0.030, dr=0.16, K=80, nscan=25, step=5)}


def spec(setup):
    s = dict(SETUPS[setup]); s["Y"] = int(round(L_BOX / s["dr"])); s["M"] = s["Y"] // s["K"]
    assert s["Y"] == s["M"] * s["K"], setup
    return s


def parse_sample(name):
    rest, setup = name[len("poly_abtem"):], "a20"
    if "_" in rest:
        rest, setup = rest.split("_", 1)
    return (3 if rest == "" else int(rest)), setup


def suffix(setup):
    return "" if setup == "a20" else f"_{setup}"
