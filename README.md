# README — Rumour Interventions on PHEME

This repository contains a full, reproducible pipeline to parse the **PHEME** rumour dataset, build temporal diffusion cascades and an auxiliary user–user network, calibrate diffusion parameters, run **counterfactual fact-checking interventions** under two models (Independent Cascade & Threshold), analyze robustness, and perform statistical tests.

---

## 1) What’s in here (top-level scripts)

| Step | Script                               | Purpose                                                               | Main Inputs                                | Key Outputs                                                                                |
| ---- | ------------------------------------ | --------------------------------------------------------------------- | ------------------------------------------ | ------------------------------------------------------------------------------------------ |
| 0    | `00_pheme_parser.py`                 | Parse raw PHEME thread folders → flat CSVs                            | `all-rnr-annotated-threads/`               | `parsed/{tweets,edges,threads}.csv`                                                        |
| 1–2  | `01-02_preprocess.py`                | Clean/normalize; filter language & tiny threads                       | `parsed/*.csv`                             | `preprocessed/{tweets,edges,threads}_clean.csv`, preprocessing report                      |
| 3–4  | `03-04_descriptive.py`               | Build **auxiliary user→user** graph (G_user) & per-thread baselines   | `preprocessed/*.csv`                       | `preprocessed/G_user_features.csv`, `preprocessed/{thread_descriptives,growth_curves}.csv` |
| 5    | `05_calibrate_diffusion.py`          | Calibrate **IC** (\lambda_M) from observed curves                     | preprocessed CSVs                          | `calibration/calibrated_ic_parameters.json`                                                |
| 6    | `06_run_interventions.py`            | Counterfactual **IC** interventions (strategies × τ × k)              | calibrated params + preprocessed           | `experiments/intervention_results_{per_thread,aggregate}.csv`, heatmaps                    |
| 7    | `07_analyze_results.py`              | Build baselines and deltas vs. no-intervention                        | step 6 outputs                             | `experiments/baseline_no_intervention.csv`, deltas + summary tables/figs                   |
| 8    | `08_analyze_decisions.py`            | “Decision matrices” & benefit-per-seed views                          | step 6–7 outputs                           | `experiments/decisions/*`, LaTeX tables                                                    |
| 9a   | `09a_calibrate_threshold.py`         | Calibrate **Threshold** (\theta_M)                                    | preprocessed CSVs                          | `calibration/calibrated_threshold_parameters.json`                                         |
| 9b   | `09b_run_interventions_threshold.py` | Counterfactual **Threshold** interventions                            | step 9a outputs                            | `experiments_threshold/intervention_threshold_{per_thread,aggregate}.csv`                  |
| 10   | `10_robustness.py`                   | IC vs TH comparison & cost-effectiveness                              | step 6–9 outputs                           | `experiments/robustness/*` figs including `ic_vs_threshold_effect.png`                     |
| 11   | `11_statistical_analysis.py`         | Statistical tests, OLS/slopes, clustered FE, heterogeneity            | steps 6–9 + step 7 baseline                | `experiments/stats/*` CSVs, LaTeX stubs, FE summaries                                      |
| 99   | `99_analyze_network.py`              | Global network analysis of (G_user) (rich-club, CCDFs, k-cores, etc.) | `preprocessed/G_user_features.csv` + edges | `network_analysis/` metrics, tables, figs                                                  |

> **Folder conventions**
>
> * `parsed/` = raw parser outputs
> * `preprocessed/` = cleaned data used everywhere else
> * `calibration/` = calibrated model parameters (IC/TH)
> * `experiments/` & `experiments_threshold/` = intervention results
> * `network_analysis/` = structural diagnostics of (G_user)

---

## 2) Environment & setup

**Python**: 3.10+ recommended (tested on 3.12)

**Install**:

```bash
# Windows PowerShell (recommended) or bash
python -m venv .venv
. .venv/Scripts/Activate.ps1    # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -U pip wheel
```

**Core dependencies** :

```
pandas numpy networkx tqdm scipy statsmodels matplotlib python-louvain
```

* `python-louvain` is required for Louvain communities.
* If you see locale or encoding issues on Windows, set `PYTHONUTF8=1`.

**Data layout**:

```
all-rnr-annotated-threads/           # raw PHEME (events/rumours/non-rumours)
parsed/{tweets,edges,threads}.csv    # created by Step 0
preprocessed/*_clean.csv             # created by Step 1–2
```

---

## 3) Running the full pipeline (recommended order)

1. **Parse raw PHEME**

```bash
python 00_pheme_parser.py \
  --root all-rnr-annotated-threads \
  --out parsed
```

*Creates `parsed/tweets.csv`, `parsed/edges.csv`, `parsed/threads.csv`.*

2. **Preprocess & filter**

```bash
python 01-02_preprocess.py
```

* English only; drop threads with < **3** tweets; timestamp/label normalization.
* Produces `preprocessed/{tweets,edges,threads}_clean.csv`.

3. **Auxiliary network & descriptives**

```bash
python 03-04_descriptive.py
```

* Builds directed weighted (G_user) (edge weights = observed interaction counts across threads).
* Computes user features: in/out degree/strength, PageRank, betweenness (on 2-core LCC of undirected projection with length = (1/w)), Louvain communities, participation coefficient.
* Saves `preprocessed/G_user_features.csv`; plus `preprocessed/thread_descriptives.csv`, `preprocessed/growth_curves.csv`.

4. **Calibrate IC**

```bash
python 05_calibrate_diffusion.py
```

* Grid search (\lambda_M) per thread vs observed cumulative curve, summarize by veracity.
* Saves `calibration/calibrated_ic_parameters.json`.

5. **Run interventions (IC)**

```bash
python 06_run_interventions.py
```

* Strategies = `earliest`, `hubs`, `bridges`, `community`, `random`
* Delays (\tau \in {0,15,30,60}), budgets (k \in {1,3,5,10}), `N_SIM=50`.
* Outputs per-thread/aggregate + figs in `experiments/`.

6. **Analyze & build baselines (IC)**

```bash
python 07_analyze_results.py
```

* Builds `experiments/baseline_no_intervention.csv` with columns:

  * `auc_M_base`, `M_final_base`, `peak_M_base`, `peak_t_base`, `runs_base`
* Creates deltas and summaries; heatmaps go to `experiments/figs/`.

7. **Decision matrices (IC)**

```bash
python 08_analyze_decisions.py
```

* Produces `experiments/decisions/` (decision matrices, BPS plots) and LaTeX tables.

8. **Calibrate TH & run interventions**

```bash
python 09a_calibrate_threshold.py
python 09b_run_interventions_threshold.py
```

* Saves `calibration/calibrated_threshold_parameters.json` and `experiments_threshold/*`.

9. **Robustness & IC-vs-TH comparison**

```bash
python 10_robustness.py
```

* Creates `experiments/robustness/*` (e.g., `ic_vs_threshold_effect.png`, `ic_bps_vs_tau.png`).

10. **Statistical analysis (IC & TH)**

```bash
python 11_statistical_analysis.py
```

* Requires: Step 6/7/9 outputs.
* Writes `experiments/stats/`:

  * `ic_per_thread_with_deltas.csv`, `th_per_thread_with_deltas.csv`
  * `ic_within_config_tests.csv` (1-sample t-tests vs 0)
  * `ic_strategy_vs_random_contrasts.csv` (paired vs random)
  * `ic_ols_trends.csv` (∆ vs τ slopes)
  * `fe_cluster_ic_dAUC.txt`, `fe_cluster_ic_dM.txt` (cluster-robust FE results)
  * Heterogeneity tables by size/duration, BPS, and LaTeX stubs for top configs.

11. *(Optional)* **Global network analysis**

```bash
python 99_analyze_network.py
```

* Prunes leaf-sinks (in-deg=1, out-deg=0) for structural diagnostics, computes metrics (density, reciprocity, clustering, assortativity, k-core sizes, rich-club (\phi(k)), etc.)
* Outputs `network_analysis/metrics_overview.json`, tables, and figures.

---

## 4) Configuration knobs (where to look)

* **Interventions (IC):** in `06_run_interventions.py`

  * `TAU_MINUTES`, `SEED_BUDGETS`, `STRATEGIES`, `N_SIM`, `LAM_F_FACTOR`
* **Calibration grids:**

  * IC: `05_calibrate_diffusion.py` (`LAMBDA_GRID`, `N_SIM`, `TIME_BIN_MIN`)
  * TH: `09a_calibrate_threshold.py` (`THETA_GRID`, `N_SIM`, `TIME_BIN_MIN`)
* **Network features:** `03-04_descriptive.py`

  * Betweenness sample size `k`, 2-core pruning, community algorithm (Louvain w/ fallback).
* **Statistical analysis:** `11_statistical_analysis.py`

  * Which tests to run, CSV/LaTeX output paths.

---

## 5) Performance notes

* **Parsing (Step 0)** is I/O bound; you can pass `--workers` if implemented; otherwise leave at 1 for Windows stability.
* **Betweenness** on large graphs can be slow; we:

  * Project to undirected, set lengths (= 1/w), **prune to 2-core**, take LCC, and **sample** sources (parameter `k`) — all implemented in `03-04_descriptive.py`.
* **IC simulations**: `N_SIM=50` is a good balance; increase only if you need tighter CIs.
* **Caching**: Step 6 prebuilds per-thread arrays once (fast cache). Avoid editing that function unless necessary.

---

## 6) Common issues & troubleshooting

* **Self-loop error on k-core** (NetworkX):
  If you see `NetworkXNotImplemented: Input graph has self loops`, ensure we **remove self-loops** before `nx.k_core`: this is already handled in `03-04_descriptive.py` (but if you customize, add:

  ```python
  UG_for_btw.remove_edges_from(nx.selfloop_edges(UG_for_btw))
  ```

  )
* **Statistical analysis baseline columns**:
  `11_statistical_analysis.py` expects the baseline file (Step 7) to have **`auc_M_base`** and **`M_final_base`** columns. If you ran an older Step 7, re-run it.
* **MixedLM dtype issues**:
  Windows CSVs sometimes create object dtypes; the script coerces numerics. If you add new columns, keep them numeric or update `to_numeric(...)` in the script.

---

## 7) Reproducibility

* We seed numpy/python RNGs in intervention scripts (`RNG_SEED=42`) for repeatable runs.
* Each step logs a **STARTED**/**COMPLETE** banner and writes summary lines (counts, timings).
* All derived products (CSVs, figs, LaTeX tables) are versioned by step folders.

---

## 8) Interpreting the main results (short takeaway)

* **Timing dominates**: At (\tau=0), most strategies strongly reduce misinformation prevalence ((\Delta \mathrm{AUC}*M)) and final adopters ((\Delta M*{\text{final}})); by (\tau\ge 15) min, gains collapse across the board.
* **Who to target**: **Hubs** and **community-aware** seeding are most reliable when you have a small or moderate budget. **Earliest** can be very strong at (k{=}1) if you can respond immediately.
* **Models agree**: **IC** and **Threshold** yield similar orderings and delay-decay shape; differences shrink at larger (\tau).

For full statistical detail (tests vs 0, paired vs random, OLS, clustered FE, heterogeneity) see `experiments/stats/`.

---


## 9) Quickstart (minimal)

```bash
# 1) Parse + preprocess
python 00_pheme_parser.py
python 01-02_preprocess.py

# 2) Build features + baselines
python 03-04_descriptive.py

# 3) Calibrate & run IC
python 05_calibrate_diffusion.py
python 06_run_interventions.py
python 07_analyze_results.py
python 08_analyze_decisions.py

# 4) Threshold & robustness
python 09a_calibrate_threshold.py
python 09b_run_interventions_threshold.py
python 10_robustness.py

# 5) Statistics
python 11_statistical_analysis.py

# 6) (Optional) Global network analysis
python 99_analyze_network.py
```

That’s it — you now have calibrated models, intervention results, figures, decision tables, and statistical evidence ready for the report and slides.
