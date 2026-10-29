"""Reproduce the report's numerical comparisons from saved results (no inference).

Run from any directory. Add --refresh-learning-curves to execute test_results.ipynb
and copy its four PDFs into report_draft/figures. See report_draft/README.md.
"""

import argparse
import ast
import json
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUNS = Path(__file__).resolve().parent
ROOT = RUNS.parents[1]
REPORT = ROOT / "report_draft"
LABELS = {
    "RANDOM_SAMPLING": "Random Sampling (Baseline)",
    "COARSE_UNCERTAINTY_MEASURE": "Coarse UM",
    "COARSE_ENTROPY_MEASURE": "Coarse EM",
    "FINE_UNCERTAINTY_MEASURE": "Fine UM",
    "FINE_ENTROPY_MEASURE": "Fine EM",
    "GROUPED_FINE_ENTROPY": "Grouped Fine EM",
    "WEIGHTED_UNCERTAINTY_MEASURE": "Weighted UM",
    "WEIGHTED_ENTROPY_MEASURE": "Weighted EM",
}
COLORS = ["#333333", "#E69F00", "#D55E00", "#56B4E9", "#0072B2",
          "#009E73", "#CC79A7", "#7B3294"]
SEEDS = {100, 200, 300, 400, 500}
GRANULARITIES = ["G_high", "G_mid", "G_low"]
METRICS = [f"{g}/{m}" for m in ("test_f1_event", "test_f1_element", "test_edit")
           for g in GRANULARITIES] + ["test_f1_class_agnostic"]
BASELINE = "RANDOM_SAMPLING"
FIGURES = ["learning-curves-mean-event-f1.pdf", "learning-curves-mean-element-f1.pdf",
           "learning-curves-edit-score.pdf", "learning-curves-class-agnostic-f1.pdf"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def load_results():
    """Validate coverage, comparable settings, and budgets reconstructed from queries."""
    training = read_json(ROOT / "src/F3Set/data/f3set-tennis/train.json")
    lookup = {clip["video"]: clip for clip in training}
    require(len(lookup) == len(training), "Duplicate training clip IDs")
    total_frames = sum(clip["num_frames"] for clip in training)
    records, acquisition_records, configurations = [], [], set()
    initial_pools = {}
    seen = set()
    for path in sorted(RUNS.glob("*/config.json")):
        config = read_json(path)
        strategy, seed = config["query_strategy"], config["seed"]
        require(strategy in LABELS and seed in SEEDS, f"Unexpected run: {path}")
        require((strategy, seed) not in seen, f"Duplicate strategy/seed: {path}")
        seen.add((strategy, seed))
        require(config["query_score_pooling"] == "MEAN", f"Not mean pooling: {path}")
        require(config["active_learning_budget_unit"] == "percent_of_training_frames",
                f"Unexpected budget unit: {path}")
        require(config["total_training_frames"] == total_frames, "Training pool mismatch")
        configurations.add(json.dumps({k: v for k, v in config.items()
                                       if k not in {"name", "seed", "query_strategy"}},
                                      sort_keys=True))
        history = read_json(path.parent / "history.json")
        require([h["round"] for h in history] == list(range(10)), "Incomplete history")
        initial = read_json(path.parent / "round_000/labeled_train.json")
        pool = {clip["video"] for clip in initial}
        require(len(pool) == len(initial) and pool <= lookup.keys(), "Invalid initial pool")
        if seed in initial_pools:
            require(pool == initial_pools[seed], "Initial pools differ within a seed")
        initial_pools[seed] = pool.copy()
        acquired_early = []
        for number, h in enumerate(history):
            round_dir = path.parent / f"round_{number:03d}"
            frames = sum(lookup[video]["num_frames"] for video in pool)
            budget = 100 * frames / total_frames
            require(frames == h["labeled_pool_frames"], f"Frame budget mismatch: {round_dir}")
            require(len(pool) == h["labeled_pool_size"], f"Clip count mismatch: {round_dir}")
            require(abs(budget - h["labeled_budget_percent"]) < 1e-10,
                    f"Reported budget mismatch: {round_dir}")
            losses = read_json(round_dir / "loss.json")
            require([x["epoch"] for x in losses] == list(range(config["f3ed"]["num_epochs"])),
                    f"Incomplete training: {round_dir}")
            eligible = [x for x in losses if x["epoch"] >= config["f3ed"]["start_val_epoch"]]
            best = max(eligible, key=lambda x: x["val_edit"])
            require(best["epoch"] == h["best_epoch"], f"Checkpoint epoch mismatch: {round_dir}")
            result = read_json(round_dir / "test_results.json")
            require(all(result[g]["test_f1_class_agnostic"] == result["test_f1_class_agnostic"]
                        for g in GRANULARITIES), "Class-agnostic scores differ by granularity")
            for metric in METRICS:
                if "/" in metric:
                    g, m = metric.split("/")
                    value = result[g][m]
                else:
                    value = result[metric]
                upper = 100 if metric.endswith("test_edit") else 1
                require(np.isfinite(value) and 0 <= value <= upper, f"Invalid score: {round_dir}")
                records.append(dict(strategy=strategy, seed=seed, round=number,
                                    budget_pct=budget, metric=metric, value=value))
            if number < 9:
                selected = read_json(round_dir / "queried_samples.json")
                require(len(set(selected)) == len(selected), "Duplicate query IDs")
                require(set(selected) <= lookup.keys() and not pool.intersection(selected),
                        f"Invalid acquisition: {round_dir}")
                if number < 4:
                    acquired_early.extend(selected)
                pool.update(selected)
        require(pool == lookup.keys(), f"Final pool is incomplete: {path}")
        clips = [lookup[video] for video in acquired_early]
        frames = sum(clip["num_frames"] for clip in clips)
        events = [event for clip in clips for event in clip["events"]]
        non_in = sum(event["label"].split("_")[-1] != "in" for event in events)
        acquisition_records.append(dict(
            strategy=strategy, seed=seed, clips=len(clips), frames=frames, events=len(events),
            events_per_1000_frames=1000 * len(events) / frames,
            mean_frames_per_clip=frames / len(clips), non_in_outcome_pct=100 * non_in / len(events),
            unique_event_classes=len({event["label"] for event in events}),
        ))
    require(seen == {(s, seed) for s in LABELS for seed in SEEDS}, "Expected 40 complete runs")
    require(len(configurations) == 1, "Incompatible experiment settings")
    rounds = pd.DataFrame(records)
    for endpoint in (0, 9):
        spread = rounds.loc[rounds["round"] == endpoint].groupby(["seed", "metric"])["value"].agg(
            lambda x: x.max() - x.min())
        require((spread < 1e-12).all(), f"Scores differ across strategies at round {endpoint}")
    return rounds, pd.DataFrame(acquisition_records)


def calculate_aulcs(rounds):
    rows = []
    for (strategy, seed, metric), curve in rounds.groupby(["strategy", "seed", "metric"]):
        curve = curve.sort_values("round")
        x, y = curve["budget_pct"].to_numpy(), curve["value"].to_numpy()
        require((np.diff(x) > 0).all() and x[-1] == 100, "Invalid budget sequence")
        scale = 1 if metric.endswith("test_edit") else 100
        area = np.sum((y[:-1] + y[1:]) * np.diff(x) / 2)
        rows.append(dict(strategy=strategy, seed=seed, metric=metric,
                         aulc=scale * area / (x[-1] - x[0])))
    aulcs = pd.DataFrame(rows)
    summary = aulcs.groupby(["strategy", "metric"]).agg(
        mean=("aulc", "mean"), sd=("aulc", "std"), seeds=("seed", "nunique")).reset_index()
    baseline = summary.loc[summary["strategy"] == BASELINE].set_index("metric")["mean"]
    summary["delta_points"] = summary["mean"] - summary["metric"].map(baseline)
    summary["relative_gain_pct"] = 100 * (summary["mean"] / summary["metric"].map(baseline) - 1)
    paired = aulcs.merge(aulcs.loc[aulcs["strategy"] == BASELINE, ["seed", "metric", "aulc"]]
                        .rename(columns={"aulc": "baseline_aulc"}), on=["seed", "metric"],
                        validate="many_to_one")
    paired["delta_points"] = paired["aulc"] - paired["baseline_aulc"]
    pair_summary = paired.groupby(["strategy", "metric"]).agg(
        mean_delta=("delta_points", "mean"), sd_delta=("delta_points", "std"),
        seeds_positive=("delta_points", lambda x: int((x > 0).sum())),
        seeds=("seed", "nunique")).reset_index()
    return aulcs, summary, paired, pair_summary


def threshold_crossings(rounds):
    """First observed mean-curve crossing; no interpolation or smoothing."""
    reference = rounds.loc[(rounds["strategy"] == BASELINE) & (rounds["round"] == 9)]
    reference = reference.groupby("metric")["value"].mean()
    means = rounds.groupby(["strategy", "metric", "round"], as_index=False).agg(
        budget_pct=("budget_pct", "mean"), value=("value", "mean"))
    rows = []
    for (strategy, metric), curve in means.groupby(["strategy", "metric"]):
        curve = curve.sort_values("round")
        for fraction in (0.90, 0.95, 1.0):
            target = fraction * reference[metric]
            hits = curve.loc[curve["value"] >= target]
            first = hits.iloc[0] if not hits.empty else None
            rows.append(dict(strategy=strategy, metric=metric, target_fraction=fraction,
                             reference_score=reference[metric], target_score=target,
                             first_round=int(first["round"]) if first is not None else None,
                             first_budget_pct=first["budget_pct"] if first is not None else None,
                             score_at_crossing=first["value"] if first is not None else None))
    return pd.DataFrame(rows)


def check_report_tables(summary):
    """Check every printed mean, sample SD, and relative percentage to two decimals."""
    text = (REPORT / "results.tex").read_text()
    table_metrics = dict(zip(["aulc_f1_event", "aulc_f1_element", "aulc_edit", "aulc_class_agnostic"],
                            ["test_f1_event", "test_f1_element", "test_edit", "test_f1_class_agnostic"]))
    checked = 0
    for table, metric in table_metrics.items():
        blocks = [b for b in re.findall(r"\\begin\{table\}.*?\\end\{table\}", text, re.S)
                  if f"\\label{{tab:{table}}}" in b]
        require(len(blocks) == 1, f"Missing or duplicate table: {table}")
        for strategy, label in LABELS.items():
            name = "R. Sampling (Baseline)" if strategy == BASELINE else label
            lines = [line for line in blocks[0].splitlines() if line.strip().startswith(name + " ")]
            require(len(lines) == 1, f"Missing or duplicate table row: {table}/{name}")
            actual = [float(x) for x in re.findall(r"[-+]?\d+\.\d{2}", lines[0])]
            keys = [metric] if metric == "test_f1_class_agnostic" else [f"{g}/{metric}" for g in GRANULARITIES]
            expected = []
            for key in keys:
                row = summary.loc[(summary["strategy"] == strategy) & (summary["metric"] == key)].iloc[0]
                expected.extend(float(f"{row[k]:.2f}") for k in ("mean", "sd", "relative_gain_pct"))
                checked += 1
            require(actual == expected, f"Table mismatch: {table}/{name}: {actual} != {expected}")
    return checked


def check_notebook_labels():
    notebook = read_json(RUNS / "test_results.ipynb")
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    assignments = [node for node in ast.parse(source).body if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "LABELS" for t in node.targets)]
    require(len(assignments) == 1 and ast.literal_eval(assignments[0].value) == LABELS,
            "Notebook strategy labels do not match the analysis")


def plot_acquisitions(acquisitions, destination):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.1), sharey=True, layout="constrained")
    specs = [("events_per_1000_frames", "Events per 1,000 acquired frames", (0, 23)),
             ("non_in_outcome_pct", "'Shot outcome' label present (% of events)", (0, 40))]
    positions = np.arange(len(LABELS))
    for ax, (metric, label, bounds) in zip(axes, specs, strict=True):
        summary = acquisitions.groupby("strategy")[metric].agg(["mean", "std"]).reindex(LABELS)
        ax.barh(positions, summary["mean"], xerr=summary["std"], color=COLORS, height=0.65,
                error_kw={"capsize": 3, "elinewidth": 1})
        ax.set(xlabel=label, xlim=bounds, yticks=positions, yticklabels=list(LABELS.values()))
        ax.axvline(summary.loc[BASELINE, "mean"], color="0.35", linestyle="--", linewidth=1)
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def refresh_learning_curves():
    import nbformat
    from nbclient import NotebookClient

    path = RUNS / "test_results.ipynb"
    notebook = nbformat.read(path, as_version=4)
    NotebookClient(notebook, timeout=120, kernel_name="python3",
                   resources={"metadata": {"path": str(RUNS)}}).execute()
    nbformat.write(notebook, path)
    for filename in FIGURES:
        shutil.copy2(RUNS / "test_results_figures" / filename, REPORT / "figures" / filename)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-learning-curves", action="store_true")
    args = parser.parse_args()
    rounds, acquisitions = load_results()
    aulcs, summary, paired, pair_summary = calculate_aulcs(rounds)
    crossings = threshold_crossings(rounds)
    checked = check_report_tables(summary)
    check_notebook_labels()
    output = RUNS / "report_analysis"
    output.mkdir(exist_ok=True)
    for name, data in {"round_metrics": rounds, "aulc_per_seed": aulcs, "aulc_summary": summary,
                       "paired_differences": paired, "paired_summary": pair_summary,
                       "threshold_crossings": crossings, "early_acquisitions": acquisitions}.items():
        data.to_csv(output / f"{name}.csv", index=False, float_format="%.12g")
    destination = REPORT / "figures/early-acquisition-composition.pdf"
    plot_acquisitions(acquisitions, destination)
    if args.refresh_learning_curves:
        refresh_learning_curves()
    print(f"Validated 40 runs, 400 rounds, and {checked} report table cells (240 printed numbers).")
    print(f"Saved analysis CSVs to {output.relative_to(ROOT)}")
    print(f"Saved acquisition figure to {destination.relative_to(ROOT)}")
    if args.refresh_learning_curves:
        print("Executed the notebook and refreshed all four report learning-curve PDFs.")


if __name__ == "__main__":
    main()
