"""Steps 7-9: cutoff, metrics, curves, failure analysis. Reads predictions files only (no GPU).

Usage (from the repo root):
  python src/evaluate.py --config configs/base.yaml --step cutoffs   # step 7, validation only
  python src/evaluate.py --config configs/base.yaml --step test      # step 8, frozen cutoffs on test
  python src/evaluate.py --config configs/base.yaml --step calibrate # step 8 extra, Platt scaling fit on val
  python src/evaluate.py --config configs/base.yaml --step failures  # step 9, by lesion type + example picks
  python src/evaluate.py --config configs/base.yaml --step examples  # step 9, photo montage (needs the photos)

Compared (12 models, 4 families):
- single model: seeds 0-4, each alone (1 answer)          -> reported as the average of 5
- MC Dropout:   seeds 0-4, each alone with 30 dropout asks -> reported as the average of 5
- Ensemble:     5 seeds, 1 answer each
- MC Dropout + StochasticDepth: seeds 0-4, each alone, 30 asks with dropout and block skipping on
                -> reported as the average of 5 (added after the main evaluation, same protocol)
- Ensemble + MC Dropout: 5 seeds x 30 asks (kept in the csv outputs, left out of figures and printed
  summaries: it matched the Ensemble everywhere)
Per-seed answers come from results/predictions/{ensemble,ensemble_mc,mc_stochdepth}_samples.npy.

- Step 7 (validation): cost-based defer cutoffs. A photo goes home only if its % says healthy,
  entropy <= t1 and MI <= t2; everything else sees a doctor. (t1, t2) is the cheapest pair on
  validation for each model and each miss cost (10, 50, 100). Saved frozen to results/cutoffs.json.
- Step 8 (test): sensitivity, AUC, ECE, Brier, risk-coverage and sensitivity-coverage curves,
  reliability diagrams, timing.
- Step 9 (test): uncertainty and errors by true lesion type, example photos.
All figures are saved to figures/.
"""
import argparse
import json
import textwrap
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from uncertainty import uncertainty

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PRED_DIR = Path("results/predictions")
FIG_DIR = Path("figures")
T1_QUANTILES = np.linspace(0, 1, 41)  # entropy cutoffs: every 2.5% of the validation spread
T2_QUANTILES = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0]  # MI cutoffs (plus "off")
RULES = ["no_defer", "entropy_only", "entropy_and_mi"]
N_BOOT = 1000          # bootstrap resamples of test lesions for 95% intervals
COVERAGE = np.linspace(1.0, 0.2, 81)  # share of photos the model keeps (rest go to a doctor)

# Figures: one fixed color per family (validated colorblind-safe), plus line style and marker
# as a second cue so identity never relies on color alone.
FAMILIES = ["single_model", "mc_dropout", "mc_stochdepth", "ensemble"]
LABEL = {"single_model": "Single model (avg of 5)", "mc_dropout": "MC Dropout (avg of 5)",
         "mc_stochdepth": "MC Dropout + StochDepth (avg of 5)", "ensemble": "Deep Ensemble",
         "ensemble_mc": "Ensemble + MC Dropout"}
COLOR = {"single_model": "#2a78d6", "mc_dropout": "#eb6834", "ensemble": "#1baf7a", "mc_stochdepth": "#eda100"}
STYLE = {"single_model": "-", "mc_dropout": "--", "ensemble": "-", "mc_stochdepth": ":"}
MARKER = {"single_model": "o", "mc_dropout": "s", "ensemble": "^", "mc_stochdepth": "D"}
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
DX_NAME = {"mel": "Melanoma", "nv": "Mole (nevus)", "bkl": "Benign keratosis", "bcc": "Basal cell carcinoma",
           "akiec": "Actinic keratosis", "vasc": "Vascular lesion", "df": "Dermatofibroma"}
EXAMPLE_DIR = Path("data/examples")  # example photos copied from the pod (not in git)


def family(name):
    """single_seed3 -> single_model, mc_seed3 -> mc_dropout, mcsd_seed3 -> mc_stochdepth."""
    if name.startswith("single_seed"):
        return "single_model"
    if name.startswith("mc_seed"):
        return "mc_dropout"
    if name.startswith("mcsd_seed"):
        return "mc_stochdepth"
    return name


def build_models(cfg):
    """One table per compared model (val + test rows): p_melanoma, entropy, mutual_info."""
    meta = pd.read_csv(PRED_DIR / "ensemble.csv")[["image_id", "lesion_id", "dx", "label", "split"]]
    ens = np.load(PRED_DIR / "ensemble_samples.npy")        # (N, 5): column i = seed i
    ens_mc = np.load(PRED_DIR / "ensemble_mc_samples.npy")  # (N, 150): seed i = columns 30i..30i+29
    sd_file = PRED_DIR / "mc_stochdepth_samples.npy"        # same layout, dropout + stochastic depth
    mc_sd = np.load(sd_file) if sd_file.exists() else None
    passes = cfg["uncertainty"]["mc_passes"]
    samples = {}
    for i, seed in enumerate(cfg["uncertainty"]["ensemble_seeds"]):
        samples[f"single_seed{seed}"] = ens[:, i:i + 1]
        samples[f"mc_seed{seed}"] = ens_mc[:, i * passes:(i + 1) * passes]
        if mc_sd is not None:
            samples[f"mcsd_seed{seed}"] = mc_sd[:, i * passes:(i + 1) * passes]
    samples["ensemble"] = ens
    samples["ensemble_mc"] = ens_mc
    models = {}
    for name, s in samples.items():
        table = meta.copy()
        table["p_melanoma"], table["entropy"], table["mutual_info"] = uncertainty(s.astype(np.float64))
        models[name] = table
    return models


def send_home(table, t1, t2):
    """The defer rule: home only if the % says healthy and both trust checks pass."""
    return (table["p_melanoma"] < 0.5) & (table["entropy"] <= t1) & (table["mutual_info"] <= t2)


def outcome(table, home, cost_miss, cost_visit):
    """Bill for one rule: missed melanomas (sent home) and doctor visits (everyone else)."""
    melanoma = table["label"] == 1
    missed = int((home & melanoma).sum())
    visits = int((~home).sum())
    return {
        "sent_home": int(home.sum()),
        "sent_home_%": round(100 * home.mean(), 1),
        "doctor_visits": visits,
        "missed_melanoma": missed,
        "melanoma_caught_%": round(100 * (1 - missed / melanoma.sum()), 1),
        "cost": cost_miss * missed + cost_visit * visits,
    }


def best_cutoffs(table, cost_miss, cost_visit, use_mi):
    """Cheapest (t1, t2) on the grid; ties go to fewer doctor visits, then to MI off."""
    t1_grid = np.unique(np.quantile(table["entropy"], T1_QUANTILES))
    t2_grid = np.unique(np.quantile(table["mutual_info"], T2_QUANTILES)) if use_mi else []
    t2_grid = [np.inf] + list(t2_grid)  # inf = MI check off, tried first so it wins ties
    best = None
    for t1 in t1_grid:
        for t2 in t2_grid:
            result = outcome(table, send_home(table, t1, t2), cost_miss, cost_visit)
            key = (result["cost"], result["doctor_visits"])
            if best is None or key < best[0]:
                best = (key, float(t1), float(t2), result)
    _, t1, t2, result = best
    return {"t1": t1, "t2": t2 if np.isfinite(t2) else None, **result}


def step_cutoffs(cfg):
    """Step 7: choose and freeze the cutoffs on validation only."""
    e = cfg["evaluate"]
    rows, frozen = [], {}
    for name, table in build_models(cfg).items():
        val = table[table["split"] == "val"].reset_index(drop=True)
        for cost_miss in e["cost_miss_sweep"]:
            base = {"model": name, "family": family(name), "cost_miss": cost_miss}
            everyone = outcome(val, pd.Series(False, index=val.index), cost_miss, e["cost_defer"])
            no_defer = outcome(val, val["p_melanoma"] < 0.5, cost_miss, e["cost_defer"])
            rows.append({**base, "rule": "everyone_to_doctor", "t1": None, "t2": None, **everyone})
            rows.append({**base, "rule": "no_defer", "t1": None, "t2": None, **no_defer})
            for rule, use_mi in [("entropy_only", False), ("entropy_and_mi", True)]:
                chosen = best_cutoffs(val, cost_miss, e["cost_defer"], use_mi)
                rows.append({**base, "rule": rule, **chosen})
                frozen.setdefault(name, {}).setdefault(str(cost_miss), {})[rule] = {
                    "t1": chosen["t1"], "t2": chosen["t2"]}

    table = pd.DataFrame(rows)
    table.to_csv("results/val_costs.csv", index=False)
    pd.DataFrame([{"model": n, "family": family(n),
                   "mean_entropy": t[t["split"] == "val"]["entropy"].mean(),
                   "mean_mi": t[t["split"] == "val"]["mutual_info"].mean()} for n, t in build_models(cfg).items()]
                 ).round(5).to_csv("results/val_uncertainty.csv", index=False)
    json.dump(frozen, open("results/cutoffs.json", "w"), indent=2)

    main = table[table["rule"] == "entropy_and_mi"]
    summary = main.groupby(["cost_miss", "family"], sort=False).agg(
        models=("model", "size"),
        cost=("cost", "mean"), cost_min=("cost", "min"), cost_max=("cost", "max"),
        sent_home_pct=("sent_home_%", "mean"), melanoma_caught_pct=("melanoma_caught_%", "mean"),
        mi_used=("t2", lambda s: int(s.notna().sum())),
    ).round(1)
    with pd.option_context("display.width", 200):
        print(summary.to_string())
    print("saved results/val_costs.csv, results/cutoffs.json")


# ---------------------------------------------------------------- step 8: test

def ece(p, y, bins):
    """Expected calibration error of the melanoma %: average gap between predicted % and the
    real melanoma share, over equal-width bins, weighted by how many photos fall in each bin."""
    idx = np.clip(np.digitize(p, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1)
    return sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean())
               for b in range(bins) if (idx == b).any())


def classification_metrics(table, bins):
    p, y = table["p_melanoma"].values, table["label"].values
    healthy = y == 0
    return {
        "auc": roc_auc_score(y, p),
        "ece": ece(p, y, bins),
        "brier": np.mean((p - y) ** 2),
        "sensitivity_at_50%": np.mean(p[~healthy] >= 0.5),
        "specificity_at_50%": np.mean(p[healthy] < 0.5),
        "healthy_above_95%_sure": np.mean(p[healthy] < 0.05),  # focal-loss "easy healthy" check
    }


def bill(home, y, cost_miss, cost_visit):
    """Per-1,000-patient cost, % melanoma caught and % sent home for one set of photos."""
    melanoma = y == 1
    missed = (home & melanoma).sum()
    return {
        "cost_per_1000": 1000 * (cost_miss * missed + cost_visit * (~home).sum()) / len(y),
        "melanoma_caught_%": 100 * (1 - missed / melanoma.sum()),
        "sent_home_%": 100 * home.mean(),
    }


def selective_curves(table):
    """Defer the least-trusted photos first (highest entropy); for each share kept, the error
    rate among kept photos (risk) and the share of all melanomas not sent home (sensitivity)."""
    order = np.argsort(table["entropy"].values, kind="stable")
    y = table["label"].values[order].astype(bool)
    called_melanoma = table["p_melanoma"].values[order] >= 0.5
    risk, sens = [], []
    for c in COVERAGE:
        k = int(round(c * len(y)))
        risk.append(np.mean(y[:k] != called_melanoma[:k]))
        sens.append(1 - (y[:k] & ~called_melanoma[:k]).sum() / y.sum())
    return np.array(risk), np.array(sens)


def style(ax, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=9)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=10)


def save(fig, name):
    FIG_DIR.mkdir(exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(FIG_DIR / name, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_curves(curves):
    deferred = 100 * (1 - COVERAGE)
    for key, ylabel, fname, title in [
        ("sens", "Melanomas caught (%)", "sensitivity_coverage.png",
         "Melanomas caught vs. share sent to a doctor (test)"),
        ("risk", "Error rate among photos kept (%)", "risk_coverage.png",
         "Errors left vs. share sent to a doctor (test)"),
    ]:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for fam in FAMILIES:
            y = 100 * curves[fam][key]
            ax.plot(deferred, y, STYLE[fam], color=COLOR[fam], linewidth=2, label=LABEL[fam],
                    marker=MARKER[fam], markevery=10, markersize=6, markeredgecolor=SURFACE)
        style(ax, "Sent to a dermatologist, least-trusted first (%)", ylabel)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.legend(frameon=False, fontsize=9, labelcolor=INK_2)
        save(fig, fname)


def plot_reliability(models, bins, fname="reliability.png"):
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True)
    for ax, fam in zip(axes, FAMILIES):
        pooled = pd.concat([t for n, t in models.items() if family(n) == fam])
        p, y = pooled["p_melanoma"].values, pooled["label"].values
        idx = np.clip(np.digitize(p, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1)
        used = [b for b in range(bins) if (idx == b).sum() >= 5]
        ax.plot([0, 100], [0, 100], color=AXIS, linewidth=1, linestyle="--")
        ax.plot([100 * p[idx == b].mean() for b in used], [100 * y[idx == b].mean() for b in used],
                STYLE[fam], color=COLOR[fam], linewidth=2, marker=MARKER[fam], markersize=6,
                markeredgecolor=SURFACE)
        style(ax, "Predicted melanoma %", "Real melanoma share (%)" if fam == FAMILIES[0] else "")
        ax.set_title(f"{LABEL[fam]}\nECE {ece(p, y, bins):.3f}", color=INK, fontsize=10, loc="left")
        ax.set_xlim(0, 100), ax.set_ylim(0, 100)
    save(fig, fname)


def plot_costs(summary, cost_sweep):
    main = summary[summary["rule"] == "entropy_and_mi"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    width = 0.2
    for i, fam in enumerate(FAMILIES):
        rows = main[main["family"] == fam].set_index("cost_miss").loc[cost_sweep]
        x = np.arange(len(cost_sweep)) + (i - 1.5) * width
        err = [rows["cost_per_1000"] - rows["cost_per_1000_lo"], rows["cost_per_1000_hi"] - rows["cost_per_1000"]]
        ax.bar(x, rows["cost_per_1000"], width - 0.03, color=COLOR[fam], label=LABEL[fam],
               yerr=err, error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 3})
    ax.axhline(1000, color=MUTED, linewidth=1, linestyle="--")
    ax.text(-0.45, 1015, "everyone to a doctor = 1,000", color=MUTED, fontsize=8, ha="left", va="bottom")
    ax.set_xticks(np.arange(len(cost_sweep)), [f"miss = {c}" for c in cost_sweep])
    style(ax, "Cost of a missed melanoma (doctor visit = 1)", "Cost per 1,000 patients")
    ax.set_title("Defer policy cost on test, 95% bootstrap intervals", color=INK, fontsize=11, loc="left", pad=40)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, ncol=2, loc="lower left",
              bbox_to_anchor=(0, 1.0), handlelength=1.2, columnspacing=1.0)
    ax.set_ylim(0, 1.05 * main["cost_per_1000_hi"].max())
    save(fig, "policy_cost.png")


def step_test(cfg):
    """Step 8: apply the frozen cutoffs once to the test photos and measure everything."""
    e = cfg["evaluate"]
    cost_sweep = e["cost_miss_sweep"]
    cutoffs = json.load(open("results/cutoffs.json"))
    models = {n: t[t["split"] == "test"].reset_index(drop=True) for n, t in build_models(cfg).items()}
    y = next(iter(models.values()))["label"].values

    # who goes home, per model x miss cost x rule (cutoffs from validation, untouched)
    home = {}
    for name, t in models.items():
        for c in cost_sweep:
            for rule in RULES:
                if rule == "no_defer":
                    home[name, c, rule] = (t["p_melanoma"] < 0.5).values
                else:
                    cut = cutoffs[name][str(c)][rule]
                    t2 = np.inf if cut["t2"] is None else cut["t2"]
                    home[name, c, rule] = send_home(t, cut["t1"], t2).values

    # point estimates, then 95% intervals from resampling whole lesions (photos of one lesion stay together)
    point = pd.DataFrame([{"model": n, "family": family(n), "cost_miss": c, "rule": r,
                           **bill(h, y, c, e["cost_defer"])} for (n, c, r), h in home.items()])
    point.to_csv("results/test_policy.csv", index=False)
    groups = list(next(iter(models.values())).groupby("lesion_id").indices.values())
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(N_BOOT):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        rows = pd.DataFrame([{"family": family(n), "cost_miss": c, "rule": r,
                              **bill(h[idx], y[idx], c, e["cost_defer"])} for (n, c, r), h in home.items()])
        fam = rows.groupby(["family", "cost_miss", "rule"]).mean()
        diff = fam.xs("ensemble", level="family") - fam.xs("single_model", level="family")
        boot.append((fam, diff))
    stats = ["cost_per_1000", "melanoma_caught_%", "sent_home_%"]
    fam_point = point.groupby(["family", "cost_miss", "rule"])[stats].mean()
    lo = pd.concat([b[0] for b in boot]).groupby(level=[0, 1, 2]).quantile(0.025)
    hi = pd.concat([b[0] for b in boot]).groupby(level=[0, 1, 2]).quantile(0.975)
    summary = fam_point.join(lo.add_suffix("_lo")).join(hi.add_suffix("_hi")).reset_index()
    summary["family"] = pd.Categorical(summary["family"], FAMILIES + ["ensemble_mc"], ordered=True)
    summary = summary.sort_values(["rule", "cost_miss", "family"]).round(1)
    summary.to_csv("results/test_summary.csv", index=False)
    d = pd.concat([b[1] for b in boot])["cost_per_1000"].groupby(level=[0, 1])
    d_point = (fam_point.xs("ensemble", level="family") - fam_point.xs("single_model", level="family"))["cost_per_1000"]
    ens_gain = pd.DataFrame({"ensemble_minus_single": d_point, "lo": d.quantile(0.025), "hi": d.quantile(0.975)}).round(1)
    ens_gain.to_csv("results/test_ensemble_vs_single.csv")

    # ranking, honesty of the %, focal-loss check
    metrics = pd.DataFrame([{"model": n, "family": family(n), **classification_metrics(t, e["ece_bins"])}
                            for n, t in models.items()])
    metrics.to_csv("results/test_metrics.csv", index=False)
    fam_metrics = metrics.drop(columns="model").groupby("family").agg(["mean", "min", "max"]).loc[FAMILIES]

    # compute cost: training epochs (GPU minutes at one model's solo speed) and prediction time
    runs = [json.load(open(f"results/seed{s}/metrics.json")) for s in cfg["uncertainty"]["ensemble_seeds"]]
    solo = pd.read_csv("results/seed0/history.csv")["seconds"].iloc[1:].median()  # s/epoch, one model alone
    pred = json.load(open(PRED_DIR / "timing.json"))
    epochs = [r["epochs_run"] for r in runs]
    passes, n_seeds = cfg["uncertainty"]["mc_passes"], len(runs)
    rows = {  # prediction seconds are for one model (or the whole ensemble), all 3,015 val + test photos
        "single_model": (1, np.mean(epochs), 1, pred["baseline"], pred["gpu"]),
        "mc_dropout": (1, np.mean(epochs), passes, pred["mc_dropout"], pred["gpu"]),
        "ensemble": (5, sum(epochs), 5, pred["ensemble"], pred["gpu"]),
        "ensemble_mc": (5, sum(epochs), 5 * passes, pred["ensemble_mc"], pred["gpu"]),
    }
    if "mc_stochdepth" in pred:
        rows["mc_stochdepth"] = (1, np.mean(epochs), passes, pred["mc_stochdepth"] / n_seeds,
                                 pred.get("gpu_mc_stochdepth", pred["gpu"]))
    timing = pd.DataFrame(rows, index=["models_trained", "train_epochs", "answers_per_photo",
                                       "predict_seconds_3015_photos", "predict_gpu"]).T.infer_objects()
    timing["train_gpu_minutes"] = (timing["train_epochs"] * solo / 60).round(1)
    timing.to_csv("results/test_timing.csv")

    curves = {}
    for fam in FAMILIES:
        per_model = [selective_curves(t) for n, t in models.items() if family(n) == fam]
        curves[fam] = {"risk": np.mean([r for r, _ in per_model], axis=0),
                       "sens": np.mean([s for _, s in per_model], axis=0)}
    pd.DataFrame({"deferred_%": 100 * (1 - COVERAGE),
                  **{f"{fam}_{k}": curves[fam][k] for fam in FAMILIES for k in ("sens", "risk")}}
                 ).round(4).to_csv("results/test_curves.csv", index=False)
    plot_curves(curves)
    plot_reliability(models, e["ece_bins"])
    plot_costs(summary, cost_sweep)

    with pd.option_context("display.width", 220, "display.max_columns", 30):
        cols = ["family", "cost_miss", "cost_per_1000", "cost_per_1000_lo", "cost_per_1000_hi",
                "melanoma_caught_%", "sent_home_%"]
        for rule in RULES:
            print(f"\n== defer policy on test: {rule} ==")
            print(summary[summary["rule"] == rule][cols].to_string(index=False))
        print("\n== ensemble minus single model, cost per 1,000 (negative = ensemble cheaper) ==")
        print(ens_gain.xs("entropy_and_mi", level="rule").to_string())
        print("\n== classification (mean over a family's models) ==")
        print(fam_metrics.xs("mean", axis=1, level=1).round(3).to_string())
        print("\n== compute ==")
        print(timing.to_string())
    print("\nsaved results/test_*.csv and figures/*.png")


# ---------------------------------------------------------------- step 9: failure analysis

def plot_by_type(fam):
    """Per lesion type: share sent to a doctor (1:100 policy) and mean entropy, single vs ensemble."""
    shown = ["single_model", "ensemble"]
    order = fam.xs("ensemble").sort_values("photos").index
    names = [f"{DX_NAME[d]} (n={int(fam.loc[('ensemble', d), 'photos'])})" for d in order]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, col, xlabel in [(axes[0], "to_doctor_%", "Sent to a doctor, 1:100 policy (%)"),
                            (axes[1], "mean_entropy", "Mean predictive entropy (nats)")]:
        y = np.arange(len(order))
        for i, f in enumerate(shown):
            ax.barh(y + (i - 0.5) * 0.38, fam.xs(f).loc[order, col], 0.34, color=COLOR[f], label=LABEL[f])
        style(ax, xlabel, "")
        ax.set_yticks(y, names)
        ax.grid(axis="y", visible=False)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="lower right")
    fig.suptitle("Which lesion types does the model doubt? (test)", color=INK, fontsize=11, x=0.01, ha="left")
    save(fig, "by_lesion_type.png")


def step_failures(cfg):
    """Step 9: uncertainty and errors by true lesion type, error detection, example photos."""
    e = cfg["evaluate"]
    cost = max(e["cost_miss_sweep"])  # the main setting, 1:100
    cutoffs = json.load(open("results/cutoffs.json"))
    models = {n: t[t["split"] == "test"].reset_index(drop=True) for n, t in build_models(cfg).items()}

    rows, detect = [], []
    for name, t in models.items():
        cut = cutoffs[name][str(cost)]["entropy_and_mi"]
        home = send_home(t, cut["t1"], np.inf if cut["t2"] is None else cut["t2"])
        d = t.assign(to_doctor=~home, called_melanoma=t["p_melanoma"] >= 0.5)
        g = d.groupby("dx").agg(
            photos=("image_id", "size"),
            mean_melanoma_pct=("p_melanoma", lambda s: 100 * s.mean()),
            mean_entropy=("entropy", "mean"), mean_mi=("mutual_info", "mean"),
            **{"called_melanoma_%": ("called_melanoma", lambda s: 100 * s.mean()),
               "to_doctor_%": ("to_doctor", lambda s: 100 * s.mean())})
        rows.append(g.assign(model=name, family=family(name)).reset_index())
        # does uncertainty point at the mistakes? AUROC of each score for "this photo is misclassified at 50%"
        wrong = (d["called_melanoma"].astype(int) != d["label"]).values
        detect.append({"model": name, "family": family(name), "errors": int(wrong.sum()),
                       "auroc_entropy": roc_auc_score(wrong, t["entropy"]),
                       "auroc_mi": roc_auc_score(wrong, t["mutual_info"]) if t["mutual_info"].max() > 0 else np.nan})
    by_type = pd.concat(rows)
    by_type.to_csv("results/test_by_lesion_type.csv", index=False)

    # hidden confident mistakes: how low a melanoma % does each model give to real melanomas?
    misses = []
    for name, t in build_models(cfg).items():
        for split in ("val", "test"):
            p = 100 * t[(t["split"] == split) & (t["label"] == 1)]["p_melanoma"]
            misses.append({"model": name, "family": family(name), "split": split, "melanomas": len(p),
                           "lowest_melanoma_%": round(p.min(), 2), "below_10%": int((p < 10).sum()),
                           "below_20%": int((p < 20).sum())})
    misses = pd.DataFrame(misses)
    misses.to_csv("results/confident_misses.csv", index=False)
    fam = by_type.groupby(["family", "dx"]).mean(numeric_only=True)
    detect = pd.DataFrame(detect)
    detect.to_csv("results/test_error_detection.csv", index=False)
    plot_by_type(fam)

    # example photos, chosen by rule from the ensemble's test answers
    ens = models["ensemble"].copy()
    singles = np.column_stack([models[f"single_seed{s}"]["p_melanoma"] for s in cfg["uncertainty"]["ensemble_seeds"]])
    ens["lowest_single_%"], ens["highest_single_%"] = 100 * singles.min(1), 100 * singles.max(1)
    picks = [
        ("Confident and right: melanoma", ens[ens["label"] == 1].nlargest(1, "p_melanoma")),
        ("Confident and right: healthy", ens[ens["label"] == 0].nsmallest(1, "entropy")),
        ("Most uncertain: sent to doctor", ens.nlargest(1, "entropy")),
        ("Models disagree most", ens.nlargest(1, "mutual_info")),
        ("Confident false alarm", ens[ens["label"] == 0].nlargest(1, "p_melanoma")),
        ("Hardest melanoma", ens[ens["label"] == 1].nsmallest(1, "p_melanoma")),
        ("One model sure it's healthy", ens[ens["label"] == 1].nsmallest(1, "lowest_single_%")),
    ]
    examples = pd.concat([row.assign(case=case) for case, row in picks]).drop_duplicates("image_id")
    cols = ["case", "image_id", "dx", "label", "p_melanoma", "entropy", "mutual_info", "lowest_single_%", "highest_single_%"]
    examples[cols].round(4).to_csv("results/examples.csv", index=False)

    with pd.option_context("display.width", 220, "display.max_columns", 30):
        for col in ["to_doctor_%", "called_melanoma_%", "mean_entropy", "mean_mi"]:
            print(f"\n== {col} by true lesion type (1:100 policy where relevant) ==")
            print(fam[col].unstack("family")[FAMILIES].assign(
                photos=fam.xs("ensemble")["photos"]).round(3).to_string())
        print("\n== does uncertainty point at mistakes? AUROC (0.5 = no, 1 = perfectly) ==")
        print(detect.groupby("family")[["errors", "auroc_entropy", "auroc_mi"]].mean().loc[FAMILIES].round(3).to_string())
        print("\n== lowest melanoma % given to a real melanoma (hidden confident mistakes) ==")
        print(misses.drop(columns="family").to_string(index=False))
        print("\n== example photos ==")
        print(examples[cols].round(3).to_string(index=False))
    print("\nsaved results/test_by_lesion_type.csv, test_error_detection.csv, examples.csv, figures/by_lesion_type.png")


def step_examples(cfg):
    """Montage of the example photos (from data/examples/ or the dataset folders)."""
    examples = pd.read_csv("results/examples.csv")
    fig, axes = plt.subplots(1, len(examples), figsize=(2.3 * len(examples), 3.6))
    for ax, (_, r) in zip(axes, examples.iterrows()):
        photo = EXAMPLE_DIR / f"{r['image_id']}.jpg"
        if not photo.exists():  # fall back to the full dataset folders
            photo = next(Path(cfg["data"]["root"]).glob(f"HAM10000_images_part_*/{r['image_id']}.jpg"))
        ax.imshow(plt.imread(photo))
        ax.set_xticks([]), ax.set_yticks([])
        for side in ax.spines.values():
            side.set_color(AXIS)
        ax.set_title(textwrap.fill(r["case"], 20), color=INK, fontsize=8.5, loc="left")
        ax.set_xlabel(f"True: {DX_NAME[r['dx']]}\nEnsemble: {100 * r['p_melanoma']:.0f}% melanoma\n"
                      f"5 models: {r['lowest_single_%']:.0f}-{r['highest_single_%']:.0f}%",
                      color=INK_2, fontsize=8, loc="left")
    save(fig, "examples.png")
    print("saved figures/examples.png")


# ---------------------------------------------------------------- step 8 extra: calibration

def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def step_calibrate(cfg):
    """Platt scaling: per model, fit calibrated % = sigmoid(a * logit(%) + b) on validation only,
    then measure on test whether the % became honest. Ranking (AUC) cannot change since a > 0."""
    bins = cfg["evaluate"]["ece_bins"]
    rows, params, calibrated = [], {}, {}
    for name, table in build_models(cfg).items():
        val, test = table[table["split"] == "val"], table[table["split"] == "test"].reset_index(drop=True)
        platt = LogisticRegression(C=1e6).fit(logit(val["p_melanoma"].values), val["label"].values)
        params[name] = {"a": float(platt.coef_[0, 0]), "b": float(platt.intercept_[0])}
        p_cal = platt.predict_proba(logit(test["p_melanoma"].values))[:, 1]
        calibrated[name] = test.assign(p_melanoma=p_cal)
        before, after = classification_metrics(test, bins), classification_metrics(calibrated[name], bins)
        rows.append({"model": name, "family": family(name), "a": params[name]["a"], "b": params[name]["b"],
                     "ece_before": before["ece"], "ece_after": after["ece"],
                     "brier_before": before["brier"], "brier_after": after["brier"],
                     "auc_before": before["auc"], "auc_after": after["auc"],
                     "healthy_above_95%_sure_after": after["healthy_above_95%_sure"]})
    table = pd.DataFrame(rows)
    table.to_csv("results/test_calibration.csv", index=False)
    json.dump(params, open("results/platt_params.json", "w"), indent=2)
    plot_reliability(calibrated, bins, "reliability_calibrated.png")
    with pd.option_context("display.width", 200):
        print(table.drop(columns="model").groupby("family").mean().loc[FAMILIES].round(3).to_string())
    print("saved results/test_calibration.csv, results/platt_params.json, figures/reliability_calibrated.png")


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved predictions (no GPU needed)")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--step", choices=["cutoffs", "test", "calibrate", "failures", "examples"], required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.step == "cutoffs":
        step_cutoffs(cfg)
    elif args.step == "test":
        step_test(cfg)
    elif args.step == "calibrate":
        step_calibrate(cfg)
    elif args.step == "failures":
        step_failures(cfg)
    elif args.step == "examples":
        step_examples(cfg)


if __name__ == "__main__":
    main()
