"""Reproducible synthetic benchmark for graph-based student group formation.

This script creates synthetic (not real student) skill vectors and a weighted
social graph, compares four grouping approaches, and writes tables and figures.

Run:
    python replicate_experiment.py

Dependencies: numpy, pandas, matplotlib, networkx
All random generators use fixed seeds for reproducibility.
"""
from __future__ import annotations

import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from pathlib import Path
from time import perf_counter
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

N = 30
K = 6
GROUP_SIZE = N // K
N_REPLICATES = 30
RANDOM_BASELINE_TRIALS = 20
BASE_SEED = 20261016
ALPHA_MAIN = 0.50
ALPHAS = [0.20, 0.35, 0.50, 0.65, 0.80]
SKILL_NAMES = ["Programming", "Writing", "Data Analysis"]


def generate_problem(seed: int):
    """Generate one fully synthetic problem instance.

    Skills: three continuous scores in [0, 1]. Each student has a randomly
    chosen primary strength, with modest noise in all dimensions.

    Social graph: an undirected weighted stochastic block model with six
    latent communities of five students. Edges are more likely within a latent
    community than between communities. This is a stylized assumption, not
    evidence about real student friendships.
    """
    rng = np.random.default_rng(seed)

    primary = rng.integers(0, len(SKILL_NAMES), size=N)
    skills = rng.uniform(0.28, 0.52, size=(N, len(SKILL_NAMES)))
    skills[np.arange(N), primary] += rng.uniform(0.25, 0.43, size=N)
    skills += rng.normal(0.0, 0.07, size=skills.shape)
    skills = np.clip(skills, 0.0, 1.0)

    latent_community = np.repeat(np.arange(K), GROUP_SIZE)
    rng.shuffle(latent_community)
    adjacency = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(i + 1, N):
            p = 0.62 if latent_community[i] == latent_community[j] else 0.11
            if rng.random() < p:
                w = rng.uniform(0.20, 1.00)
                adjacency[i, j] = w
                adjacency[j, i] = w

    return skills, adjacency, primary, latent_community


def metrics(labels: np.ndarray, skills: np.ndarray, adjacency: np.ndarray):
    """Calculate cohesion C and group-mean imbalance B for a complete partition.

    C = total edge weight kept within groups / total edge weight.
    B = between-group skill sum-of-squares / total skill sum-of-squares.
        B is 0 when every group mean equals the whole-class mean; larger values
        mean that groups' average profiles differ more from the class average.
        This is a balance metric, NOT a within-group diversity metric.
    """
    total_w = float(np.triu(adjacency, k=1).sum())
    if total_w > 0:
        same_group = labels[:, None] == labels[None, :]
        internal_w = float(np.triu(adjacency * same_group, k=1).sum())
        cohesion = internal_w / total_w
    else:
        cohesion = 0.0

    class_mean = skills.mean(axis=0)
    total_ss = float(((skills - class_mean) ** 2).sum())
    if total_ss <= 1e-12:
        imbalance = 0.0
    else:
        between_ss = 0.0
        for g in range(K):
            ix = np.where(labels == g)[0]
            if len(ix):
                group_mean = skills[ix].mean(axis=0)
                between_ss += len(ix) * float(((group_mean - class_mean) ** 2).sum())
        imbalance = between_ss / total_ss
    return float(cohesion), float(imbalance)


def composite_score(labels: np.ndarray, skills: np.ndarray, adjacency: np.ndarray, alpha: float):
    c, b = metrics(labels, skills, adjacency)
    return alpha * c - (1.0 - alpha) * b


def partial_score(labels: np.ndarray, skills: np.ndarray, adjacency: np.ndarray, alpha: float):
    """Score a partial assignment, ignoring graph edges with unassigned nodes."""
    assigned = labels >= 0
    if not np.any(assigned):
        return 0.0
    total_w = float(np.triu(adjacency, k=1).sum())
    same = labels[:, None] == labels[None, :]
    assigned_pair = assigned[:, None] & assigned[None, :]
    internal_assigned = float(np.triu(adjacency * same * assigned_pair, k=1).sum())
    c_partial = internal_assigned / total_w if total_w > 0 else 0.0

    class_mean = skills.mean(axis=0)
    total_ss = float(((skills - class_mean) ** 2).sum())
    between_partial = 0.0
    for g in range(K):
        ix = np.where(labels == g)[0]
        if len(ix):
            group_mean = skills[ix].mean(axis=0)
            between_partial += len(ix) * float(((group_mean - class_mean) ** 2).sum())
    b_partial = between_partial / total_ss if total_ss > 1e-12 else 0.0
    return alpha * c_partial - (1.0 - alpha) * b_partial


def greedy_initial_partition(skills: np.ndarray, adjacency: np.ndarray, alpha: float, rng: np.random.Generator):
    """Create a capacity-constrained greedy assignment for a chosen alpha."""
    degree = adjacency.sum(axis=1)
    class_mean = skills.mean(axis=0)
    skill_dist = np.linalg.norm(skills - class_mean, axis=1)

    if alpha >= 0.999:
        order = list(np.argsort(-degree, kind="stable"))
    elif alpha <= 0.001:
        order = list(np.argsort(-skill_dist, kind="stable"))
    else:
        # Rank aggregation: prioritize students who have both strong social
        # connectivity and skill profiles far from the class mean.
        degree_order = np.argsort(-degree, kind="stable")
        skill_order = np.argsort(-skill_dist, kind="stable")
        degree_rank = np.empty(N, dtype=int)
        skill_rank = np.empty(N, dtype=int)
        degree_rank[degree_order] = np.arange(N)
        skill_rank[skill_order] = np.arange(N)
        combined_rank = degree_rank + skill_rank
        order = list(np.argsort(combined_rank, kind="stable"))

    labels = np.full(N, -1, dtype=int)
    counts = np.zeros(K, dtype=int)
    # Randomize only exact assignment-score ties for reproducibility without
    # systematically favoring group 0.
    for student in order:
        feasible = [g for g in range(K) if counts[g] < GROUP_SIZE]
        scores = []
        for g in feasible:
            trial = labels.copy()
            trial[student] = g
            scores.append(partial_score(trial, skills, adjacency, alpha))
        best = max(scores)
        tied = [g for g, score in zip(feasible, scores) if abs(score - best) <= 1e-12]
        chosen = tied[int(rng.integers(0, len(tied)))]
        labels[student] = chosen
        counts[chosen] += 1
    return labels


def local_search(labels: np.ndarray, skills: np.ndarray, adjacency: np.ndarray, alpha: float):
    """Best-improvement pairwise swaps with incremental objective deltas."""
    labels = labels.copy()
    total_w = float(np.triu(adjacency, k=1).sum())
    class_mean = skills.mean(axis=0)
    total_ss = float(((skills - class_mean) ** 2).sum())
    group_sums = np.vstack([skills[labels == g].sum(axis=0) for g in range(K)])
    group_dev = group_sums - GROUP_SIZE * class_mean
    # link_to_group[i, g] is the sum of weights from node i to current group g.
    link_to_group = np.column_stack([adjacency[:, labels == g].sum(axis=1) for g in range(K)])
    iterations = 0

    while True:
        best_gain = 1e-12
        best_swap = None
        for i in range(N):
            a = int(labels[i])
            for j in range(i + 1, N):
                b = int(labels[j])
                if a == b:
                    continue
                edge_ij = adjacency[i, j]
                delta_internal = (
                    link_to_group[i, b] - edge_ij - link_to_group[i, a]
                    + link_to_group[j, a] - edge_ij - link_to_group[j, b]
                )
                delta_c = delta_internal / total_w if total_w > 0 else 0.0

                old_a = group_dev[a]
                old_b = group_dev[b]
                new_a = old_a - skills[i] + skills[j]
                new_b = old_b - skills[j] + skills[i]
                delta_between_ss = (
                    float(new_a @ new_a) + float(new_b @ new_b)
                    - float(old_a @ old_a) - float(old_b @ old_b)
                ) / GROUP_SIZE
                delta_b = delta_between_ss / total_ss if total_ss > 1e-12 else 0.0
                gain = alpha * delta_c - (1.0 - alpha) * delta_b
                if gain > best_gain:
                    best_gain = gain
                    best_swap = (i, j, a, b, new_a, new_b)
        if best_swap is None:
            break
        i, j, a, b, new_a, new_b = best_swap
        labels[i], labels[j] = labels[j], labels[i]
        group_sums[a] += skills[j] - skills[i]
        group_sums[b] += skills[i] - skills[j]
        group_dev[a] = new_a
        group_dev[b] = new_b
        # Update each node's summed edge weight to groups a and b after swap.
        change = adjacency[:, j] - adjacency[:, i]
        link_to_group[:, a] += change
        link_to_group[:, b] -= change
        iterations += 1
        if iterations >= 1000:
            break
    return labels, iterations


def optimize_method(skills, adjacency, alpha, seed):
    rng = np.random.default_rng(seed)
    start = perf_counter()
    initial = greedy_initial_partition(skills, adjacency, alpha, rng)
    improved, swap_count = local_search(initial, skills, adjacency, alpha)
    elapsed = perf_counter() - start
    c, b = metrics(improved, skills, adjacency)
    return improved, c, b, alpha * c - (1 - alpha) * b, elapsed, swap_count


def random_partition(rng: np.random.Generator):
    order = rng.permutation(N)
    labels = np.empty(N, dtype=int)
    for g in range(K):
        labels[order[g * GROUP_SIZE:(g + 1) * GROUP_SIZE]] = g
    return labels


def group_size_variance(labels):
    sizes = np.bincount(labels, minlength=K)
    return float(np.var(sizes)), sizes.tolist()


def make_graph_figure(skills, adjacency, primary):
    G = nx.Graph()
    for i in range(N):
        G.add_node(i, primary=int(primary[i]))
    for i in range(N):
        for j in range(i + 1, N):
            if adjacency[i, j] > 0:
                G.add_edge(i, j, weight=float(adjacency[i, j]))
    pos = nx.spring_layout(G, seed=123, weight="weight")
    plt.figure(figsize=(10, 7))
    node_colors = [primary[i] for i in range(N)]
    widths = [0.35 + 1.6 * G[u][v]["weight"] for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, alpha=0.28, width=widths, edge_color="grey")
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=plt.cm.Set2,
                           node_size=430, edgecolors="black", linewidths=0.5)
    nx.draw_networkx_labels(G, pos, labels={i: f"S{i+1:02d}" for i in range(N)}, font_size=7)
    plt.title("Synthetic social graph (node colors indicate strongest synthetic skill)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUT / "figure1_synthetic_network.png", dpi=220, bbox_inches="tight")
    plt.close()
    return G, pos


def make_partition_figure(G, pos, partitions, skills):
    titles = ["Balanced random", "Social-only", "Skill-balance-only", "Proposed (α = 0.50)"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (name, labels), title in zip(axes.flat, partitions.items(), titles):
        node_colors = [int(labels[i]) for i in range(N)]
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.20, width=0.6, edge_color="grey")
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, cmap=plt.cm.tab10,
                               node_size=270, edgecolors="black", linewidths=0.4)
        nx.draw_networkx_labels(G, pos, labels={i: f"{i+1}" for i in range(N)}, font_size=6, ax=ax)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.suptitle("Four partitions of the same synthetic network\n(node color shows assigned group)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "figure2_partition_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_results_figure(summary: pd.DataFrame):
    order = ["Random", "Social-only", "Skill-only", "Proposed α=0.50"]
    summary = summary.set_index("method").loc[order].reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(summary))
    axes[0].bar(x, summary["cohesion_mean"], yerr=summary["cohesion_sd"], capsize=4)
    axes[0].set_xticks(x, ["Random", "Social-only", "Skill-only", "Proposed"], rotation=20)
    axes[0].set_ylabel("Social cohesion C (higher is better)")
    axes[0].set_title("Internal edge-weight fraction")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, summary["imbalance_mean"], yerr=summary["imbalance_sd"], capsize=4)
    axes[1].set_xticks(x, ["Random", "Social-only", "Skill-only", "Proposed"], rotation=20)
    axes[1].set_ylabel("Skill imbalance B (lower is better)")
    axes[1].set_title("Between-group skill variance fraction")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "figure3_method_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_sensitivity_figure(sensitivity: pd.DataFrame):
    grouped = sensitivity.groupby("alpha").agg(
        cohesion_mean=("cohesion", "mean"), cohesion_sd=("cohesion", "std"),
        imbalance_mean=("imbalance", "mean"), imbalance_sd=("imbalance", "std")
    ).reset_index()
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.errorbar(grouped["alpha"], grouped["cohesion_mean"], yerr=grouped["cohesion_sd"],
                 marker="o", capsize=3, label="Social cohesion C")
    ax1.set_xlabel("Social weight α")
    ax1.set_ylabel("Social cohesion C")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.errorbar(grouped["alpha"], grouped["imbalance_mean"], yerr=grouped["imbalance_sd"],
                 marker="s", capsize=3, linestyle="--", label="Skill imbalance B")
    ax2.set_ylabel("Skill imbalance B")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    plt.title("Sensitivity to the social-cohesion weight")
    fig.tight_layout()
    fig.savefig(OUT / "figure4_alpha_sensitivity.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    all_rows = []
    sensitivity_rows = []
    first_example = None

    for rep in range(N_REPLICATES):
        seed = BASE_SEED + rep
        skills, adjacency, primary, latent_community = generate_problem(seed)
        if rep == 0:
            first_example = (skills, adjacency, primary, latent_community)

        # Balanced random baseline: average 20 independent balanced assignments
        random_rows = []
        for trial in range(RANDOM_BASELINE_TRIALS):
            labels = random_partition(np.random.default_rng(seed * 1000 + trial))
            c, b = metrics(labels, skills, adjacency)
            random_rows.append((c, b, labels))
        random_c = float(np.mean([r[0] for r in random_rows]))
        random_b = float(np.mean([r[1] for r in random_rows]))
        random_labels = random_rows[0][2]
        all_rows.append({
            "replicate": rep + 1, "seed": seed, "method": "Random",
            "cohesion": random_c, "imbalance": random_b, "objective": np.nan,
            "runtime_seconds": np.nan, "accepted_swaps": 0,
            "group_size_variance": group_size_variance(random_labels)[0],
            "group_sizes": str(group_size_variance(random_labels)[1])
        })

        methods = [
            ("Social-only", 1.0),
            ("Skill-only", 0.0),
            ("Proposed α=0.50", ALPHA_MAIN),
        ]
        best_labels_by_method = {"Random": random_labels}
        for name, alpha in methods:
            labels, c, b, obj, elapsed, swaps = optimize_method(
                skills, adjacency, alpha, seed=seed + int(alpha * 10000) + 777
            )
            best_labels_by_method[name] = labels
            all_rows.append({
                "replicate": rep + 1, "seed": seed, "method": name,
                "cohesion": c, "imbalance": b, "objective": obj,
                "runtime_seconds": elapsed, "accepted_swaps": swaps,
                "group_size_variance": group_size_variance(labels)[0],
                "group_sizes": str(group_size_variance(labels)[1])
            })

        for alpha in ALPHAS:
            labels, c, b, obj, elapsed, swaps = optimize_method(
                skills, adjacency, alpha, seed=seed + int(alpha * 10000) + 1777
            )
            sensitivity_rows.append({
                "replicate": rep + 1, "seed": seed, "alpha": alpha,
                "cohesion": c, "imbalance": b, "objective": obj,
                "runtime_seconds": elapsed, "accepted_swaps": swaps
            })

        if rep == 0:
            first_example_partitions = best_labels_by_method

    by_instance = pd.DataFrame(all_rows)
    by_instance.to_csv(OUT / "results_by_instance.csv", index=False)
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(OUT / "alpha_sensitivity_by_instance.csv", index=False)

    # Aggregate using standard deviation across the 30 independently generated
    # synthetic instances. The Random value within each instance is itself an
    # average over 20 random partitions.
    summary = by_instance.groupby("method").agg(
        cohesion_mean=("cohesion", "mean"), cohesion_sd=("cohesion", "std"),
        imbalance_mean=("imbalance", "mean"), imbalance_sd=("imbalance", "std"),
        runtime_median_seconds=("runtime_seconds", "median"),
        swaps_mean=("accepted_swaps", "mean"),
        max_group_size_variance=("group_size_variance", "max")
    ).reset_index()
    method_order = ["Random", "Social-only", "Skill-only", "Proposed α=0.50"]
    summary["method"] = pd.Categorical(summary["method"], categories=method_order, ordered=True)
    summary = summary.sort_values("method").reset_index(drop=True)
    summary["method"] = summary["method"].astype(str)
    summary.to_csv(OUT / "results_summary.csv", index=False)

    sensitivity_summary = sensitivity.groupby("alpha").agg(
        cohesion_mean=("cohesion", "mean"), cohesion_sd=("cohesion", "std"),
        imbalance_mean=("imbalance", "mean"), imbalance_sd=("imbalance", "std"),
        objective_mean=("objective", "mean"), objective_sd=("objective", "std")
    ).reset_index()
    sensitivity_summary.to_csv(OUT / "alpha_sensitivity_summary.csv", index=False)

    # Export first synthetic instance and its edges so the example network can
    # be independently inspected and reused without regenerating it.
    skills, adjacency, primary, latent_community = first_example
    students = pd.DataFrame({
        "student_id": [f"S{i+1:02d}" for i in range(N)],
        "programming": skills[:, 0], "writing": skills[:, 1],
        "data_analysis": skills[:, 2],
        "strongest_synthetic_skill": [SKILL_NAMES[i] for i in primary],
        "latent_social_community_for_generation_only": latent_community + 1
    })
    students.to_csv(OUT / "example_synthetic_students.csv", index=False)
    edge_rows = []
    for i in range(N):
        for j in range(i + 1, N):
            if adjacency[i, j] > 0:
                edge_rows.append({"source": f"S{i+1:02d}", "target": f"S{j+1:02d}", "weight": adjacency[i, j]})
    pd.DataFrame(edge_rows, columns=["source", "target", "weight"]).to_csv(OUT / "example_synthetic_edges.csv", index=False)

    # Example partitions file for the first replicate. These are algorithm
    # outputs for synthetic data and are not student recommendations.
    parts = first_example_partitions
    assignment_data = {"student_id": [f"S{i+1:02d}" for i in range(N)]}
    for name, labels in parts.items():
        assignment_data[name.lower().replace(" ", "_").replace("α", "alpha").replace("=", "")] = [int(x + 1) for x in labels]
    pd.DataFrame(assignment_data).to_csv(OUT / "example_partitions.csv", index=False)

    G, pos = make_graph_figure(skills, adjacency, primary)
    make_partition_figure(G, pos, parts, skills)
    make_results_figure(summary)
    make_sensitivity_figure(sensitivity)

    config = {
        "synthetic_data_only": True, "n_students": N, "n_groups": K,
        "group_size": GROUP_SIZE, "replicates": N_REPLICATES,
        "random_partitions_per_replicate": RANDOM_BASELINE_TRIALS,
        "base_seed": BASE_SEED, "main_alpha": ALPHA_MAIN,
        "sensitivity_alphas": ALPHAS,
        "edge_probability_within_latent_community": 0.62,
        "edge_probability_between_latent_communities": 0.11,
        "edge_weight_range_if_present": [0.20, 1.00],
        "metric_definitions": {
            "cohesion_C": "weight of edges internal to assigned groups divided by total graph edge weight",
            "imbalance_B": "between-group sum of squares of skill-profile means divided by total skill sum of squares; lower is better",
            "composite_J": "alpha*C - (1-alpha)*B"
        }
    }
    with open(OUT / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("Synthetic experiment complete.")
    print(f"Outputs: {OUT}")
    print("\nSummary (means across 30 synthetic instances):")
    print(summary.to_string(index=False, formatters={
        "cohesion_mean": "{:.4f}".format, "cohesion_sd": "{:.4f}".format,
        "imbalance_mean": "{:.4f}".format, "imbalance_sd": "{:.4f}".format,
        "runtime_median_seconds": (lambda x: "NA" if pd.isna(x) else f"{x:.4f}"),
        "swaps_mean": "{:.2f}".format,
        "max_group_size_variance": "{:.4f}".format
    }))
    print("\nSensitivity summary:")
    print(sensitivity_summary.to_string(index=False, formatters={
        "alpha": "{:.2f}".format, "cohesion_mean": "{:.4f}".format,
        "cohesion_sd": "{:.4f}".format, "imbalance_mean": "{:.4f}".format,
        "imbalance_sd": "{:.4f}".format, "objective_mean": "{:.4f}".format,
        "objective_sd": "{:.4f}".format
    }))


if __name__ == "__main__":
    main()
