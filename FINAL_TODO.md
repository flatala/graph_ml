# Final Implementation TODO

## Quick Summary

You need to modify 4 notebooks. Here's what to do for each:

---

## 1. ✅ evolve_gcn.ipynb
**Status**: Already has multi-seed training
**Action**: Nothing needed (or re-run if you want)

---

## 2. 🔄 static_gcn.ipynb
**Goal**: Add multi-seed training + feature reduction

**Quick steps**:
1. Add `import random` to imports
2. Add feature reduction function from `SIMPLIFIED_IMPLEMENTATION_GUIDE.md`
3. Add multi-seed config: `SEEDS = [42, 123, 456]`
4. Use reduced features in builder with `feature_cols=kept_features`
5. Change cache to `'../../graph_cache_reduced_features_fixed'`
6. Wrap training loop in `for seed in SEEDS:`
7. Save results to `../../results/static_gcn_multi_seed/`

**Full code**: See `MULTI_SEED_IMPLEMENTATION_GUIDE.md` Section 1

---

## 3. 🔄 graph_features_baseline.ipynb
**Goal**: Add enhanced graph features + feature reduction + result saving

**Quick steps**:
1. Add feature reduction function
2. Replace `compute_graph_features()` with enhanced version (7 features instead of 3)
   - Total degree, in-degree, out-degree
   - PageRank
   - Betweenness (approximated)
   - Degree ratio
   - Normalized degree
3. Use reduced features in builder with `feature_cols=kept_features`
4. Change cache to `'../../graph_cache_reduced_features_fixed'`
5. At end, add results saving code (from `SIMPLIFIED_IMPLEMENTATION_GUIDE.md` Section 2)
6. Save to `../../results/graph_features_baseline/`

**Full code**: See `SIMPLIFIED_IMPLEMENTATION_GUIDE.md` Section 2

---

## 4. 🔄 baselines.ipynb
**Goal**: Add result saving only (already has feature reduction!)

**Quick steps**:
1. At the very end of notebook, add results saving code
2. Save each model to separate directory:
   - `../../results/baselines/logistic_regression/`
   - `../../results/baselines/random_forest/`
   - `../../results/baselines/xgboost/`
3. Create `all_metrics.csv` and `summary_statistics.csv` for each

**Full code**: See `SIMPLIFIED_IMPLEMENTATION_GUIDE.md` Section 3

---

## 5. 📊 results_comparison.ipynb
**Goal**: Create unified comparison plots

**Quick steps**:
1. Use the template from `results_comparison_template.ipynb`
2. Update the loading section to handle both multi-seed and single-run formats
3. Run after all other notebooks complete

**Full code**: See `SIMPLIFIED_IMPLEMENTATION_GUIDE.md` Section 4

---

## Execution Order

1. Run `static_gcn.ipynb` (will take time - 3 seeds × 4 K values)
2. Run `graph_features_baseline.ipynb` (faster - single run)
3. Run `baselines.ipynb` (faster - single run, 3 models)
4. Run `results_comparison.ipynb` (fast - just loads and plots)

---

## Expected Results

### Multi-seed models (with error bars):
- Temporal GCN
- Static GCN

### Single-run models (no error bars):
- MLP + Graph Features (with 7 graph structural features!)
- Logistic Regression
- Random Forest
- XGBoost

### Plots will show:
1. F1 vs K (all models, GNNs with error bars)
2. AUC vs K (all models, GNNs with error bars)
3. Heatmap of F1 scores
4. Best model per K
5. Model stability analysis

---

## Key Files to Reference

1. **SIMPLIFIED_IMPLEMENTATION_GUIDE.md** - Complete code snippets
2. **results_comparison_template.ipynb** - Ready-to-use comparison notebook
3. **MULTI_SEED_IMPLEMENTATION_GUIDE.md** - Detailed static_gcn modifications

---

## Pro Tips

- Test with fewer epochs first to make sure everything works
- The cache directory will be created automatically on first run
- Results are saved incrementally, so you can stop/restart
- All models use the same 36 reduced features (except MLP gets +7 graph features)

Good luck! 🚀
