# Study Group Formation Research Project
## Equal-Size Student Group Formation with Social Cohesion and Skill Balance

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Domain: Graph Theory](https://img.shields.io/badge/Domain-Graph%20Theory-purple.svg)]()
[![Optimization: Heuristic%20%26%20MILP](https://img.shields.io/badge/Optimization-Greedy%20%2B%20Local%20Search%20%26%20MILP-orange.svg)]()

A comprehensive research study and algorithmic framework for partitioning a student cohort into equal-sized study groups that simultaneously maximize **within-group social cohesion** and **between-group skill balance**.

This repository contains the complete computational implementation, synthetic and real-world experimental benchmarks, mathematical programming models (MILP), visual figures, and the academic research manuscript.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Mathematical Formulation](#mathematical-formulation)
  - [Graph and Skill Representation](#graph-and-skill-representation)
  - [Evaluation Metrics](#evaluation-metrics)
  - [Composite Objective](#composite-objective)
  - [Mixed Integer Linear Programming (MILP) Formulation](#mixed-integer-linear-programming-milp-formulation)
- [Algorithms and Methods](#algorithms-and-methods)
- [Repository Structure](#repository-structure)
- [Installation and Environment Setup](#installation-and-environment-setup)
- [How to Run the Experiments](#how-to-run-the-experiments)
  - [1. Synthetic Benchmark Replication](#1-synthetic-benchmark-replication)
  - [2. Real-World Dataset & Exact Optimization Experiments](#2-real-world-dataset--exact-optimization-experiments)
- [Experimental Results and Findings](#experimental-results-and-findings)
  - [Synthetic Benchmark (30 Replicates, N=30, K=6)](#synthetic-benchmark-30-replicates-n30-k6)
  - [Sensitivity Analysis across α](#sensitivity-analysis-across-α)
  - [Real-World Benchmark (UCI Student Performance, N=15, K=3)](#real-world-benchmark-uci-student-performance-n15-k3)
  - [Solver Scalability Comparison](#solver-scalability-comparison)
- [Visualizations](#visualizations)
- [Research Transparency, Ethics, and Limitations](#research-transparency-ethics-and-limitations)
- [References](#references)

---

## Project Overview

In cooperative learning, group composition significantly impacts educational engagement and academic performance. A well-designed study group should satisfy two competing desiderata:
1. **Social Cohesion ($C$)**: Students should share interpersonal compatibility or communication affinities to foster collaborative dialogue and psychological safety.
2. **Skill Balance ($B$)**: Groups should maintain comparable average competency profiles across relevant subject domains, avoiding the creation of elite or underprivileged groups.

Random assignment often creates severe disparities in group competency profiles and fractures social clusters. Manual assignment is combinatorial and prone to cognitive bias. 

This project formulates the problem as an attributed, capacity-constrained graph partitioning task and provides:
- A fast, effective **two-phase greedy heuristic with pairwise swap local search**.
- A reproducible **synthetic benchmark** across 30 graph instances ($N=30$ students, $K=6$ groups of size 5).
- A **real-world empirical benchmark** using the UCI Student Performance Dataset ($N=649$ students).
- An exact **Mixed Integer Linear Program (MILP)** solved via HiGHS with an $L_1$ variance surrogate.
- An **exhaustive combinatorial search** establishing global ground-truth optima and demonstrating that the greedy heuristic recovers **~98.2%** of the achievable objective in fractions of a second.

---

## Mathematical Formulation

### Graph and Skill Representation

Let $V = \{1, 2, \dots, N\}$ represent the cohort of $N$ students partitioned into $K$ disjoint groups $V_1, V_2, \dots, V_K$ such that:
$$\bigcup_{k=1}^K V_k = V \quad \text{and} \quad V_a \cap V_b = \emptyset \quad (\forall a \neq b)$$

Each group has an exact capacity constraint:
$$|V_k| = \frac{N}{K} \quad \forall k \in \{1, \dots, K\}$$

- **Social Graph**: $G = (V, E, W)$, an undirected weighted graph where $w_{ij} \in [0, 1]$ represents the social compatibility or communication affinity between students $i$ and $j$.
- **Skill Vectors**: Each student $i$ is associated with a $d$-dimensional normalized skill profile $\mathbf{s}_i \in [0, 1]^d$. The cohort mean skill vector is:
  $$\bar{\mathbf{s}}_{\text{cohort}} = \frac{1}{N} \sum_{i=1}^N \mathbf{s}_i$$
  The mean skill vector of group $k$ is:
  $$\bar{\mathbf{s}}_k = \frac{1}{|V_k|} \sum_{i \in V_k} \mathbf{s}_i$$

### Evaluation Metrics

1. **Social Cohesion ($C \in [0, 1]$, Higher is Better)**:
   Measures the proportion of total graph edge weight retained within the assigned groups:
   $$C = \frac{\sum_{k=1}^K \sum_{i < j \in V_k} w_{ij}}{\sum_{i < j \in V} w_{ij}}$$

2. **Skill Imbalance ($B \ge 0$, Lower is Better)**:
   Measures the normalized between-group sum of squares of skill means relative to total cohort variance:
   $$B = \frac{\sum_{k=1}^K |V_k| \|\bar{\mathbf{s}}_k - \bar{\mathbf{s}}_{\text{cohort}}\|^2}{\sum_{i=1}^N \|\mathbf{s}_i - \bar{\mathbf{s}}_{\text{cohort}}\|^2}$$
   $B = 0$ indicates that every group's average skill profile perfectly mirrors the cohort average.

### Composite Objective

To balance social cohesion against skill imbalance, we define the scalar composite objective $J(\alpha)$:
$$J(\alpha) = \alpha C - (1 - \alpha) B$$
where $\alpha \in [0, 1]$ is a policy parameter that controls the trade-off. By default, $\alpha = 0.50$ treats both criteria with equal priority.

### Mixed Integer Linear Programming (MILP) Formulation

To benchmark against exact optimization, the problem is formulated as a MILP:
- Binary assignment variables: $x_{ik} \in \{0, 1\}$, indicating if student $i$ is assigned to group $k$.
- Continuous edge retention variables: $z_{ij} \in [0, 1]$ for each $(i, j) \in E$.
- Continuous absolute deviation variables: $d_{kt} \ge 0$ ($L_1$ linearisation of quadratic skill imbalance).

$$\max_{x, z, d} \quad \alpha \left(\frac{1}{W_{\text{tot}}} \sum_{(i,j) \in E} w_{ij} z_{ij}\right) - (1 - \alpha) \cdot \text{scale} \cdot \sum_{k=1}^K \sum_{t=1}^d d_{kt}$$

Subject to:
1. $\sum_{k=1}^K x_{ik} = 1 \quad \forall i \in V$ (unique assignment)
2. $\sum_{i=1}^N x_{ik} = \frac{N}{K} \quad \forall k \in \{1, \dots, K\}$ (strict equal capacity)
3. $z_{ij} \le 1 - x_{ik} + x_{jk}$ and $z_{ij} \le 1 + x_{ik} - x_{jk} \quad \forall (i, j) \in E, \forall k$ (edge retention logic)
4. $d_{kt} \ge \pm \left(\sum_{i=1}^N s_{it} x_{ik} - \frac{N}{K} \bar{s}_{\text{cohort}, t}\right) \quad \forall k, t$ (absolute deviation surrogate)
5. $x_{0,0} = 1$ (symmetry breaking)

---

## Algorithms and Methods

| Method | Formulation / Strategy | Characteristics |
|---|---|---|
| **Balanced Random** | Uniform random assignment with capacity enforcement ($N/K$ per group). Averaged over 20 independent trials per replicate. | Baseline representing unguided assignment. |
| **Social-Only** | Greedy assignment + Local Search with $\alpha = 1.0$. | Maximizes internal edge weight; ignores skill balance entirely. |
| **Skill-Only** | Greedy assignment + Local Search with $\alpha = 0.0$. | Minimizes between-group skill variance; ignores social relationships. |
| **Proposed Heuristic ($\alpha = 0.50$)** | **Phase 1**: Capacity-constrained greedy construction using rank aggregation (degree centrality + Euclidean distance from cohort mean).<br>**Phase 2**: Best-improvement pairwise swap local search with $O(1)$ incremental delta evaluations. | Jointly optimizes social cohesion and skill balance in under 0.1 seconds. |
| **Exact MILP (HiGHS)** | Exact mathematical program using $L_1$ balance surrogate solved via `scipy.optimize.milp`. | Finds provable optimum on surrogate model; evaluates on true $J$. |
| **Exhaustive Optimum** | Full combinatorial enumeration of non-isomorphic partitions ($252,252$ partitions for $N=15, K=3$). | Establishes absolute global ground-truth optimum for true $J$. |

---

## Repository Structure

```text
Study_Group_Research_Project/
├── .gitignore                                  # Git ignore specifications (Python, environments, OS, IDEs)
├── README.md                                   # Comprehensive project documentation
├── requirements.txt                            # Python dependencies (numpy, pandas, matplotlib, networkx, scipy, python-docx)
├── replicate_experiment.py                     # Main executable script for the synthetic benchmark (30 instances)
├── Revised_Study_Group_Formation_Report.docx   # Academic research report document
├── outputs/                                    # Generated results and figures from the synthetic benchmark
│   ├── alpha_sensitivity_by_instance.csv       # Per-instance metrics across 5 alpha settings
│   ├── alpha_sensitivity_summary.csv           # Summary statistics for alpha sensitivity
│   ├── example_partitions.csv                  # Group labels for example network under 4 methods
│   ├── example_synthetic_edges.csv             # Weighted edge list of example network (seed 20261016)
│   ├── example_synthetic_students.csv          # Student attributes and skills for example network
│   ├── experiment_config.json                  # Experiment hyperparameters and metric definitions
│   ├── figure1_synthetic_network.png           # Visual graph layout (spring layout, skill-colored nodes)
│   ├── figure2_partition_comparison.png        # 4-panel comparison of partition structures
│   ├── figure3_method_comparison.png           # Cohesion vs. Imbalance bar charts with error bars
│   ├── figure4_alpha_sensitivity.png           # Sensitivity trade-off curves across alpha values
│   ├── results_by_instance.csv                 # Detailed metrics across all 30 synthetic instances
│   └── results_summary.csv                     # Primary results summary table (mean, SD, runtime, swaps)
├── real world data set experiment files/       # Real-world dataset and exact optimization extension
│   ├── exact_and_scale.py                      # Exhaustive combinatorial search & optimality gap analysis
│   ├── fig_ip_vs_greedy.png                    # Plot comparing MILP vs. Greedy heuristic performance
│   ├── ip_experiment.py                        # HiGHS MILP implementation and benchmark runner
│   └── report_additions.md                     # Academic writeup of the real-data MILP extension
└── student+performance/                        # UCI Student Performance Dataset (Cortez & Silva, 2008)
    ├── student-mat.csv                         # Mathematics course student records (395 records)
    ├── student-por.csv                         # Portuguese language course records (649 records)
    ├── student-merge.R                         # R script for student record merging
    └── student.txt                             # Attribute descriptions and metadata documentation
```

---

## Installation and Environment Setup

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13.
- Windows, macOS, or Linux.

### Setup Instructions

1. **Clone or Navigate to the Repository**:
   ```bash
   cd Study_Group_Research_Project
   ```

2. **Create and Activate a Virtual Environment** (Recommended):
   - **Windows (PowerShell)**:
     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```
   - **macOS / Linux**:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run the Experiments

### 1. Synthetic Benchmark Replication

Runs the 30-instance stochastic benchmark ($N=30, K=6$), executes all four algorithms, computes sensitivity across $\alpha \in \{0.20, 0.35, 0.50, 0.65, 0.80\}$, and saves all tables and figures to `outputs/`:

```bash
python replicate_experiment.py
```

*Note: Execution completes in ~15-20 seconds. Output files in `outputs/` are updated automatically.*

### 2. Real-World Dataset & Exact Optimization Experiments

The real-world experiments evaluate the UCI Portuguese student dataset (`student-por.csv`):

- **Run Exhaustive Search & Exact Optimality Gap Benchmark** ($N=15, K=3$, $252,252$ partitions):
  ```bash
  python "real world data set experiment files/exact_and_scale.py"
  ```
- **Run HiGHS MILP vs. Greedy Heuristic Comparison**:
  ```bash
  python "real world data set experiment files/ip_experiment.py"
  ```

---

## Experimental Results and Findings

### Synthetic Benchmark (30 Replicates, N=30, K=6)

Averaged across 30 independent synthetic instances (Stochastic Block Model graph + 3D skill vectors, base seed `20261016`):

| Method | Cohesion $C$ (mean ± SD) | Imbalance $B$ (mean ± SD) | Median Runtime (s) | Accepted Swaps | Max Capacity Variance |
|---|---|---|---|---|---|
| **Random** (20 trials/inst) | $0.1364 \pm 0.0082$ | $0.1749 \pm 0.0104$ | — | 0.00 | 0.0000 |
| **Social-Only** ($\alpha = 1.0$) | $\mathbf{0.5174 \pm 0.0439}$ | $0.1780 \pm 0.0620$ | ~0.078 s | 10.30 | 0.0000 |
| **Skill-Only** ($\alpha = 0.0$) | $0.1493 \pm 0.0356$ | $\mathbf{0.0035 \pm 0.0013}$ | ~0.075 s | 10.40 | 0.0000 |
| **Proposed Heuristic** ($\alpha = 0.50$) | **$0.4737 \pm 0.0545$** | **$0.0442 \pm 0.0147$** | **~0.089 s** | **12.87** | **0.0000** |

**Key Takeaways**:
- The proposed heuristic captures **91.6%** of the maximum possible social cohesion achieved by the Social-Only method ($0.4737$ vs $0.5174$).
- Simultaneously, it suppresses skill imbalance by **74.7%** relative to random assignment ($0.0442$ vs $0.1749$).
- Group size capacity constraints ($N/K = 5$) are strictly satisfied in 100% of cases (variance = 0.0000).

### Sensitivity Analysis across $\alpha$

Evaluating the proposed heuristic under varying policy preferences:

| Weight $\alpha$ | Cohesion $C$ (mean ± SD) | Imbalance $B$ (mean ± SD) | Objective $J$ (mean ± SD) | Policy Implication |
|---|---|---|---|---|
| **0.20** | $0.3928 \pm 0.0446$ | $0.0129 \pm 0.0046$ | $0.0683 \pm 0.0090$ | Strong priority on balanced skills. |
| **0.35** | $0.4319 \pm 0.0539$ | $0.0220 \pm 0.0078$ | $0.1369 \pm 0.0170$ | Moderate emphasis on skills. |
| **0.50** | $0.4737 \pm 0.0545$ | $0.0442 \pm 0.0147$ | $0.2147 \pm 0.0256$ | Balanced compromise (default). |
| **0.65** | $0.4869 \pm 0.0517$ | $0.0624 \pm 0.0198$ | $0.2946 \pm 0.0321$ | Moderate emphasis on social cohesion. |
| **0.80** | $0.5068 \pm 0.0558$ | $0.1045 \pm 0.0420$ | $0.3845 \pm 0.0427$ | Strong priority on social cohesion. |

### Real-World Benchmark (UCI Student Performance, N=15, K=3)

Evaluated across 20 random classroom samples from `student-por.csv` (Cortez & Silva, 2008). Edges derived via lifestyle and demographic homophily compatibility; skills derived from prior grade, study effort, and attendance.

| Method | Cohesion $C$ | Imbalance $B$ | Objective $J$ | Mean Gap to Global Optimum | Optimal in | Runtime |
|---|---|---|---|---|---|---|
| **Random Baseline** | 0.2859 | 0.1418 | 0.0721 | — | — | < 0.001 s |
| **Greedy + Local Search** | 0.4507 | 0.0375 | **0.2066** | **1.81%** | **9 / 20** | **~0.03 s** |
| **Exact MILP (HiGHS, $L_1$)** | 0.4624 | 0.0441 | **0.2092** | **0.62%** | **14 / 20** | **~2.40 s** |
| **Exhaustive Global Optimum** | 0.4601 | 0.0393 | **0.2104** | **0.00%** | **20 / 20** | — |

**Key Takeaways**:
- The greedy local search heuristic achieves a mean optimality gap of only **1.81%** against the absolute global optimum, matching it exactly in 45% of problem instances.
- The greedy heuristic runs **~80× faster** than the MILP solver on small instances.

### Solver Scalability Comparison

Testing exact MILP (HiGHS with a 90-second cutoff) against the Greedy + Local Search heuristic across increasing problem sizes:

| Problem Size | Exact MILP (HiGHS) Status | Greedy $J$ vs. MILP $J$ | Scalability Analysis |
|---|---|---|---|
| **$N=15, K=3$** | Proven optimal in 1.6–3.1 s | Greedy equal or within 1.8% of MILP | MILP effective for small groups. |
| **$N=20, K=4$** | Time limit reached (90 s) | Greedy $0.1509$ vs. MILP $0.1568$ | MILP struggle begins; near parity. |
| **$N=30, K=6$** | **Time limit reached in 3/3 instances** | **Greedy (0.1157, 0.0952, 0.1130) beats MILP (0.1054, 0.0810, 0.0944)** | **MILP fails to converge; incumbent solutions are 9–15% worse than greedy.** |

*Conclusion*: Exact solvers fail to scale to standard classroom sizes ($N \ge 30$) under practical time budgets, confirming that the greedy local-search heuristic is both necessary and practically superior for educational deployment.

---

## Visualizations

All figures generated by `replicate_experiment.py` are saved to the `outputs/` directory:

1. **Figure 1 (`figure1_synthetic_network.png`)**: Spring layout of an example synthetic student social network ($N=30$), where node color indicates the student's primary skill dimension and edge width reflects compatibility weight.
2. **Figure 2 (`figure2_partition_comparison.png`)**: 4-panel comparison of student grouping under Random, Social-Only, Skill-Only, and the Proposed heuristic ($\alpha = 0.50$).
3. **Figure 3 (`figure3_method_comparison.png`)**: Bar charts comparing mean social cohesion $C$ and skill imbalance $B$ across all methods with $\pm 1$ SD error bars.
4. **Figure 4 (`figure4_alpha_sensitivity.png`)**: Dual-axis sensitivity curve showing the trade-off progression between cohesion and imbalance as $\alpha$ increases from $0.20$ to $0.80$.

---

## Research Transparency, Ethics, and Limitations

1. **Synthetic vs. Real Data Transparency**:
   - The primary benchmark in `replicate_experiment.py` uses **synthetic stochastic data** generated via a Stochastic Block Model. The findings are algorithmic simulations and do not represent empirical observations from a live educational trial.
2. **Derived Social Compatibility**:
   - In the real-world benchmark (`student-por.csv`), social network edges are derived via attribute homophily proxies (e.g., age, school, freetime, extracurriculars) rather than direct sociometric surveys.
3. **Algorithmic Fairness and Diversity**:
   - The metric $B$ enforces **balance between group centroids**, not diversity within individual groups. In educational settings, instructor oversight is recommended when assigning groups to avoid accidental demographic clustering.

---

## References

1. **Cortez, P., & Silva, A. (2008).** Using data mining to predict secondary school student performance. *Proceedings of 5th Future Business Technology Conference (FUBUTEC)*, pp. 5–12.
2. **Jenkins, P. R., et al. (2023).** Fair and diverse student team formation via constrained graph partitioning. *IEEE Transactions on Learning Technologies*.
3. **Jenkins, P. R., et al. (2026).** Graph-based student grouping with fairness constraints. *IEEE Signal Processing Magazine*.
4. **Lappas, T., Liu, K., & Terzi, E. (2009).** Finding a team of experts in social networks. *Proceedings of the 15th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD)*, pp. 467–476.
5. **Cao, Y., et al. (2025).** GP-DQN: Deep reinforcement learning for attribute-aware graph partitioning. *Information Sciences*.
