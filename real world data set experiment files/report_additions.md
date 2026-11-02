# Report additions: Integer-programming extension on real data

*Insert as new Section 4.7 (method), Section 5.5 (results) and updates to Sections 3, 6 and 7.*

## 4.7 Integer-programming formulation and real-data benchmark

**Dataset.** UCI Student Performance data (Cortez & Silva, 2008), Portuguese-language course file: 649 real students, 33 attributes. Each experimental "classroom" is a random sample of 15 students (3 groups of 5), repeated for 20 samples (seed 20261024).

**Skill vector** (all in [0,1]): prior academic attainment (G1/20), study effort ((studytime-1)/3) and attendance (1 - min(absences,30)/30). G2 and G3 were excluded to avoid leaking later outcomes.

**Social compatibility weights.** The dataset contains no friendship records, so w_ij is *derived* from real attributes: same school (0.25), same address type (0.10), same sex (0.10), age within 1 year (0.15), similarity in "goout" (0.20) and "freetime" (0.10), same reason for choosing the school (0.10). Pairs scoring below 0.55 receive no edge. This is a homophily proxy, not observed friendship.

**MILP.** Binary x_ik = 1 if student i is in group k; continuous z_ij in [0,1] for each edge; continuous d_kt >= 0.

maximize  alpha * (1/W_tot) * sum_{(i,j) in E} w_ij z_ij  -  (1-alpha) * s * sum_{k,t} d_kt

subject to
- sum_k x_ik = 1 for all i
- sum_i x_ik = n/K for all k
- z_ij <= 1 - x_ik + x_jk and z_ij <= 1 + x_ik - x_jk for every edge and every k (z_ij can be 1 only if i and j share a group)
- d_kt >= +/-( sum_i a_it x_ik - (n/K) mu_t ) (absolute deviation of group skill sums)
- x_{0,0} = 1 (symmetry breaking)

Because the true imbalance B is a convex *quadratic* in x, the MILP uses an L1 (absolute-deviation) surrogate so a standard MILP solver (HiGHS via scipy.optimize.milp) can be used. Every method is therefore **scored on the true J = alpha*C - (1-alpha)*B**, not the surrogate.

**Ground truth.** For n=15, K=3 there are only 252,252 distinct partitions with student 0 fixed in group 0, so the true global optimum of J was found by exhaustive enumeration. This gives exact optimality gaps for both the heuristic and the IP.

## 5.5 Results

**Table 3. Real-data benchmark, 20 samples, n=15, K=3, alpha=0.5.**

| Method | C (mean) | B (mean) | J (mean) | Mean gap to optimum | Optimal in |
|---|---|---|---|---|---|
| Random (20 partitions/instance) | 0.2859 | 0.1418 | 0.0721 | - | - |
| Greedy + local search | 0.4507 | 0.0375 | 0.2066 | 1.81% | 9 / 20 |
| IP (HiGHS, L1 surrogate) | 0.4624 | 0.0441 | 0.2092 | 0.62% | 14 / 20 |
| Exhaustive optimum | 0.4601 | 0.0393 | 0.2104 | 0 | 20 / 20 |

IP median solve time 2.4 s (max 11 s) versus about 0.03 s for the heuristic. The worst single-instance gap was 8.1% for greedy and 4.2% for the IP.

**Table 4. Scalability (HiGHS, 90 s limit, 2-3 instances per size).**

| Size | IP status | Greedy J vs IP J |
|---|---|---|
| n=15, K=3 | optimal in 1.6-3.1 s | equal in 2/3, IP lower in 1/3 (L1 surrogate) |
| n=20, K=4 | time limit reached | greedy 0.1509 vs IP 0.1568; greedy 0.1504 vs IP 0.1466 |
| n=30, K=6 | time limit reached (3/3) | greedy 0.1157/0.0952/0.1130 vs IP 0.1054/0.0810/0.0944 |

At the original problem size (30 students, 6 groups) the exact solver did not converge in 90 s and returned incumbents 9-15% worse than the heuristic, which ran in under 1 s.

## Text for Sections 3, 6, 7

- **Closing the Jenkins et al. gap (Sec. 2.3/6.2).** The report previously stated that no comparison with an integer-programming solution had been made. This extension adds an exact MILP and exhaustive-search comparison, but on a *reduced* problem (n=15) and with an L1 surrogate for B. It is still not a reproduction of the Jenkins et al. model (which also has fairness constraints and within-group diversity).
- **Finding 1.** The heuristic is within 1.8% of the true optimum on average (optimal in 45% of instances) and recovers about 98% of the achievable objective, so its quality is acceptable for small classes.
- **Finding 2.** The IP is closer to optimal (0.6%) but is not exact on the true objective because of the surrogate; replacing it with an MIQP/MIQCP solver would remove this.
- **Finding 3.** The IP does not scale: at n=30 it fails to prove optimality in 90 s and underperforms the heuristic. This justifies the heuristic for realistic class sizes and suggests a hybrid (use greedy solution as a warm start).
- **Limitations to add.** (i) Social edges are derived from attributes, not observed friendships; (ii) dataset is one Portuguese secondary-school course, so results may not transfer to university cohorts; (iii) 20 samples of n=15 and 2-3 of larger sizes is small; (iv) the 90 s limit and HiGHS are arbitrary choices; (v) the scalability test used a free solver (a commercial solver such as Gurobi/CPLEX could solve larger cases); (vi) the 2-3 rep scaling results are indicative only.
- **Ethics.** The dataset is public and anonymised, but use of attributes like sex, age and address type to build "compatibility" edges should be flagged as a fairness risk.

## Reference to add
Cortez, P., & Silva, A. (2008). Using data mining to predict secondary school student performance. *Proc. 5th FUture BUsiness TEChnology Conference (FUBUTEC)*, 5-12.
