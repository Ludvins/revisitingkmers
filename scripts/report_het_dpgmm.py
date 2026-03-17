#!/usr/bin/env python3
"""report_het_dpgmm.py

Read results/dpgmm_exploration/results.jsonl and produce a markdown report
with comparison tables for each experiment.

Usage
-----
    python scripts/report_het_dpgmm.py \\
        --results_dir results/dpgmm_exploration \\
        --output results/dpgmm_exploration/report.md
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np


# ── load results ──────────────────────────────────────────────────────────────

def load_results(results_dir):
    path = os.path.join(results_dir, "results.jsonl")
    if not os.path.exists(path):
        sys.exit(f"No results.jsonl found in {results_dir}")
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    print(f"Loaded {len(rows)} result rows from {path}")
    return rows


# ── markdown helpers ──────────────────────────────────────────────────────────

def md_table(headers, rows, fmt=None):
    """Return a GitHub-flavoured markdown table string."""
    if fmt is None:
        fmt = ["{}"] * len(headers)
    lines = []
    lines.append("| " + " | ".join(str(h) for h in headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        cells = [f.format(v) for f, v in zip(fmt, row)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def section(title, level=2):
    return "\n" + "#" * level + " " + title + "\n"


# ── Experiment A ──────────────────────────────────────────────────────────────

def report_exp_A(rows):
    data = [r for r in rows if r.get("experiment") == "A"]
    if not data:
        return "_(no Exp A results yet)_\n"

    # Group by embedder × species × sample → average over samples
    # Pivot: rows = embedder, cols = species
    from collections import defaultdict
    scores = defaultdict(lambda: defaultdict(list))  # [emb][species] -> [(f05, f09)]
    for r in data:
        emb = r["embedder"]
        sp  = r["species"]
        scores[emb][sp].append((r.get("f1_05_count", 0), r.get("f1_09_count", 0)))

    embedders = sorted(scores.keys())
    species_list = sorted({r["species"] for r in data})

    # Table 1: F1>0.5 averaged over samples per (embedder, species)
    lines = [section("Experiment A — Embedder × Species: F1>0.5 count (avg over samples)")]
    headers = ["Embedder"] + species_list + ["avg"]
    rows_t = []
    for emb in embedders:
        cells = [emb]
        vals = []
        for sp in species_list:
            if sp in scores[emb]:
                v = np.mean([x[0] for x in scores[emb][sp]])
                cells.append(f"{v:.1f}")
                vals.append(v)
            else:
                cells.append("—")
        cells.append(f"{np.mean(vals):.1f}" if vals else "—")
        rows_t.append(cells)
    lines.append(md_table(headers, rows_t))

    # Table 2: F1>0.9
    lines.append(section("Experiment A — Embedder × Species: F1>0.9 count (avg over samples)", 3))
    rows_t2 = []
    for emb in embedders:
        cells = [emb]
        vals = []
        for sp in species_list:
            if sp in scores[emb]:
                v = np.mean([x[1] for x in scores[emb][sp]])
                cells.append(f"{v:.1f}")
                vals.append(v)
            else:
                cells.append("—")
        cells.append(f"{np.mean(vals):.1f}" if vals else "—")
        rows_t2.append(cells)
    lines.append(md_table(headers, rows_t2))

    # Also show n_clusters
    lines.append(section("Experiment A — Avg number of clusters produced", 3))
    nclust = defaultdict(lambda: defaultdict(list))
    for r in data:
        nclust[r["embedder"]][r["species"]].append(r.get("n_clusters", 0))
    rows_t3 = []
    for emb in embedders:
        cells = [emb]
        for sp in species_list:
            if sp in nclust[emb]:
                v = np.mean(nclust[emb][sp])
                cells.append(f"{v:.0f}")
            else:
                cells.append("—")
        cells.append("")
        rows_t3.append(cells)
    lines.append(md_table(headers, rows_t3))

    return "\n".join(lines) + "\n"


# ── Experiment B ──────────────────────────────────────────────────────────────

def report_exp_B(rows):
    data = [r for r in rows if r.get("experiment") == "B"]
    if not data:
        return "_(no Exp B results yet)_\n"

    lines = [section("Experiment B — DPGMM Hyperparameter Sensitivity")]
    lines.append("F1>0.5 count averaged over samples 5 and 6 (reference species, ug_labeled).\n")

    # Grid: pca_dim × merge_threshold
    grid_data = [r for r in data if r.get("sweep") == "grid"]
    if grid_data:
        lines.append(section("pca_dim × merge_threshold grid (F1>0.5)", 3))
        pca_dims = sorted({r["dpgmm_params"]["pca_dim"] for r in grid_data})
        merge_ts = sorted({r["dpgmm_params"]["merge_threshold"] for r in grid_data})
        headers = ["pca_dim \\ merge_threshold"] + [str(mt) for mt in merge_ts]
        grid = defaultdict(lambda: defaultdict(list))
        for r in grid_data:
            p = r["dpgmm_params"]["pca_dim"]
            m = r["dpgmm_params"]["merge_threshold"]
            grid[p][m].append(r.get("f1_05_count", 0))
        grid_rows = []
        for p in pca_dims:
            row = [str(p)]
            for m in merge_ts:
                v = np.mean(grid[p][m]) if grid[p][m] else float("nan")
                row.append(f"{v:.1f}" if not np.isnan(v) else "—")
            grid_rows.append(row)
        lines.append(md_table(headers, grid_rows))

        # Same for F1>0.9
        lines.append(section("pca_dim × merge_threshold grid (F1>0.9)", 3))
        grid09 = defaultdict(lambda: defaultdict(list))
        for r in grid_data:
            p = r["dpgmm_params"]["pca_dim"]
            m = r["dpgmm_params"]["merge_threshold"]
            grid09[p][m].append(r.get("f1_09_count", 0))
        grid_rows09 = []
        for p in pca_dims:
            row = [str(p)]
            for m in merge_ts:
                v = np.mean(grid09[p][m]) if grid09[p][m] else float("nan")
                row.append(f"{v:.1f}" if not np.isnan(v) else "—")
            grid_rows09.append(row)
        lines.append(md_table(headers, grid_rows09))

    # n_init sweep
    ni_data = [r for r in data if r.get("sweep") == "n_init"]
    if ni_data:
        lines.append(section("n_init sweep", 3))
        ni_vals = sorted({r["dpgmm_params"]["n_init"] for r in ni_data})
        ni_rows = []
        for ni in ni_vals:
            subset = [r for r in ni_data if r["dpgmm_params"]["n_init"] == ni]
            f05 = np.mean([r.get("f1_05_count", 0) for r in subset])
            f09 = np.mean([r.get("f1_09_count", 0) for r in subset])
            ni_rows.append([str(ni), f"{f05:.1f}", f"{f09:.1f}"])
        lines.append(md_table(["n_init", "F1>0.5", "F1>0.9"], ni_rows))

    # max_components sweep
    mc_data = [r for r in data if r.get("sweep") == "max_components"]
    if mc_data:
        lines.append(section("max_components sweep", 3))
        mc_vals = sorted({r["dpgmm_params"]["max_components"] for r in mc_data})
        mc_rows = []
        for mc in mc_vals:
            subset = [r for r in mc_data if r["dpgmm_params"]["max_components"] == mc]
            f05 = np.mean([r.get("f1_05_count", 0) for r in subset])
            f09 = np.mean([r.get("f1_09_count", 0) for r in subset])
            nk  = np.mean([r.get("n_clusters", 0) for r in subset])
            mc_rows.append([str(mc), f"{f05:.1f}", f"{f09:.1f}", f"{nk:.0f}"])
        lines.append(md_table(["max_components", "F1>0.5", "F1>0.9", "avg n_clusters"],
                               mc_rows))

    return "\n".join(lines) + "\n"


# ── Experiment C ──────────────────────────────────────────────────────────────

def report_exp_C(rows):
    data = [r for r in rows if r.get("experiment") == "C"]
    if not data:
        return "_(no Exp C results yet)_\n"

    lines = [section("Experiment C — Diagonal vs Full Cluster Covariance")]
    lines.append("Averaged over samples 5 and 6 (reference species, ug_labeled).\n")

    pca_dims = sorted({r["pca_dim"] for r in data})
    cov_types = sorted({r["het_covariance_type"] for r in data})

    for metric_key, metric_label in [("f1_05_count", "F1>0.5"), ("f1_09_count", "F1>0.9")]:
        lines.append(section(f"Covariance type comparison — {metric_label}", 3))
        grid = defaultdict(lambda: defaultdict(list))
        for r in data:
            if r.get(metric_key, -1) >= 0:
                grid[r["het_covariance_type"]][r["pca_dim"]].append(r[metric_key])
        headers = ["covariance_type"] + [f"pca_dim={p}" for p in pca_dims]
        tbl_rows = []
        for ct in cov_types:
            row = [ct]
            for p in pca_dims:
                vals = grid[ct][p]
                row.append(f"{np.mean(vals):.1f}" if vals else "—")
            tbl_rows.append(row)
        lines.append(md_table(headers, tbl_rows))

    return "\n".join(lines) + "\n"


# ── Experiment D ──────────────────────────────────────────────────────────────

def report_exp_D(rows):
    data = [r for r in rows if r.get("experiment") == "D"]
    if not data:
        return "_(no Exp D results yet)_\n"

    lines = [section("Experiment D — LLA Prior Optimization Diagnostics")]

    # Convergence summary
    lines.append(section("MacKay convergence summary", 3))
    hdr = ["base_model", "hessian", "tau_final", "tau_trace_steps",
           "var_mean", "var_p50", "var_p95", "var_unassigned_corr",
           "F1>0.5", "F1>0.9"]
    tbl = []
    for r in data:
        trace = r.get("tau_trace", [])
        vs = r.get("variance_stats", {})
        tbl.append([
            r.get("base_model", "?"),
            r.get("hessian", "?"),
            f"{trace[-1]:.4f}" if trace else "—",
            str(len(trace)),
            f"{vs.get('mean', float('nan')):.2e}",
            f"{vs.get('p50', float('nan')):.2e}",
            f"{vs.get('p95', float('nan')):.2e}",
            f"{r.get('var_unassigned_corr', float('nan')):.4f}",
            str(r.get("f1_05_count", "?")),
            str(r.get("f1_09_count", "?")),
        ])
    lines.append(md_table(hdr, tbl))

    # Tau traces (compact)
    lines.append(section("MacKay τ traces (first 10 steps)", 3))
    for r in data:
        trace = r.get("tau_trace", [])
        label = f"`{r.get('base_model')}/{r.get('hessian')}`"
        short = ", ".join(f"{v:.4f}" for v in trace[:10])
        if len(trace) > 10:
            short += f", ... (converged at step {len(trace)})"
        lines.append(f"- {label}: [{short}]")

    # GGN vs EF comparison
    lines.append(section("GGN vs Empirical Fisher: variance magnitude", 3))
    by_base = defaultdict(dict)
    for r in data:
        by_base[r.get("base_model")][r.get("hessian")] = r
    for base, hess_dict in sorted(by_base.items()):
        lines.append(f"\n**{base}**\n")
        if "ggn" in hess_dict and "ef" in hess_dict:
            ggn_var = hess_dict["ggn"].get("variance_stats", {})
            ef_var  = hess_dict["ef"].get("variance_stats", {})
            tbl2 = [
                ["GGN", f"{ggn_var.get('mean', float('nan')):.3e}",
                 f"{ggn_var.get('p50', float('nan')):.3e}",
                 f"{ggn_var.get('p95', float('nan')):.3e}",
                 f"{hess_dict['ggn'].get('prior_precision_final', '?'):.4f}"],
                ["EF",  f"{ef_var.get('mean', float('nan')):.3e}",
                 f"{ef_var.get('p50', float('nan')):.3e}",
                 f"{ef_var.get('p95', float('nan')):.3e}",
                 f"{hess_dict['ef'].get('prior_precision_final', '?'):.4f}"],
            ]
            lines.append(md_table(["hessian", "var_mean", "var_p50", "var_p95", "tau"], tbl2))

    return "\n".join(lines) + "\n"


# ── Experiment E ──────────────────────────────────────────────────────────────

def report_exp_E(rows):
    data = [r for r in rows if r.get("experiment") == "E"]
    if not data:
        return "_(no Exp E results yet)_\n"

    lines = [section("Experiment E — DPGMM Convergence Diagnostics")]
    hdr = ["species", "sample", "n_iter", "converged", "n_merges",
           "merge_kl_median", "merge_kl_max", "frac_unassigned", "F1>0.5", "F1>0.9"]
    tbl = []
    for r in sorted(data, key=lambda x: (x.get("species",""), x.get("sample", 0))):
        tbl.append([
            r.get("species", "?"),
            str(r.get("sample", "?")),
            str(r.get("n_iter_best", r.get("n_iter", "?"))),
            str(r.get("converged", "?")),
            str(r.get("n_merges", "?")),
            f"{r.get('merge_kl_median', float('nan')):.3f}",
            f"{r.get('merge_kl_max', float('nan')):.3f}",
            f"{r.get('frac_unassigned', float('nan')):.3f}",
            str(r.get("f1_05_count", "?")),
            str(r.get("f1_09_count", "?")),
        ])
    lines.append(md_table(hdr, tbl))

    # Show K_active trace for first result
    if data and data[0].get("k_trace"):
        r = data[0]
        k_trace = r["k_trace"]
        lines.append(f"\n**K_active trace** ({r.get('species')}, sample {r.get('sample')}):\n")
        lines.append(f"`{k_trace[:40]}`\n")

    return "\n".join(lines) + "\n"


# ── summary ───────────────────────────────────────────────────────────────────

def report_summary(rows):
    lines = [section("Summary: Best F1>0.5 per Configuration", 2)]
    if not rows:
        return "_(no results yet)_\n"

    # Find overall best across all experiments
    best = {}
    for r in rows:
        exp = r.get("experiment", "?")
        f05 = r.get("f1_05_count", 0)
        f09 = r.get("f1_09_count", 0)
        key = (exp, r.get("embedder", r.get("base_model", "?")),
               r.get("species", "?"), r.get("sample", "?"))
        if key not in best or f05 > best[key][0]:
            best[key] = (f05, f09, r)

    lines.append("Top configurations by F1>0.5 (all experiments):\n")
    top = sorted(best.values(), key=lambda x: -x[0])[:15]
    tbl = []
    for f05, f09, r in top:
        emb = r.get("embedder", r.get("base_model", "?"))
        sp = r.get("species", "?")
        sa = r.get("sample", "?")
        exp = r.get("experiment", "?")
        params = ""
        if exp == "B":
            p = r.get("dpgmm_params", {})
            params = f"pca={p.get('pca_dim')},mt={p.get('merge_threshold')}"
        elif exp == "C":
            params = f"cov={r.get('het_covariance_type')},pca={r.get('pca_dim')}"
        elif exp == "D":
            params = f"hessian={r.get('hessian')}"
        tbl.append([exp, emb, sp, str(sa), params, str(f05), str(f09)])
    lines.append(md_table(["Exp", "Embedder", "Species", "Sample", "Params",
                            "F1>0.5", "F1>0.9"], tbl))
    return "\n".join(lines) + "\n"


# ── save flat CSV ─────────────────────────────────────────────────────────────

def save_flat_csv(rows, output_dir):
    path = os.path.join(output_dir, "results_flat.csv")
    if not rows:
        return
    # Flatten nested dicts one level
    flat_rows = []
    for r in rows:
        flat = {}
        for k, v in r.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    flat[f"{k}.{sub_k}"] = sub_v
            elif isinstance(v, list):
                flat[k] = json.dumps(v)   # serialize lists as JSON string
            else:
                flat[k] = v
        flat_rows.append(flat)
    # Collect all field names
    seen = {}
    for row in flat_rows:
        for k in row:
            seen[k] = None
    fieldnames = list(seen.keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(flat_rows)
    print(f"Saved flat CSV → {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate markdown report from het DPGMM exploration results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results_dir", default="results/dpgmm_exploration")
    p.add_argument("--output", default="results/dpgmm_exploration/report.md")
    return p.parse_args()


def main():
    args = parse_args()
    rows = load_results(args.results_dir)

    parts = [
        "# Heteroscedastic DPGMM Exploration Report\n",
        f"_Generated from `{args.results_dir}/results.jsonl` "
        f"({len(rows)} result rows)_\n",
        report_summary(rows),
        report_exp_A(rows),
        report_exp_B(rows),
        report_exp_C(rows),
        report_exp_D(rows),
        report_exp_E(rows),
    ]

    report = "\n".join(parts)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(report)
    print(f"Report written → {args.output}")

    save_flat_csv(rows, args.results_dir)


if __name__ == "__main__":
    main()
