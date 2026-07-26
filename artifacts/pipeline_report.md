# Pipeline Report — synthetic fallback dataset (make_classification-based)

_Generated 2026-07-27T01:00:51_

## 1. Dataset Summary

- **Target column:** `target`
- **Problem type:** `classification`
- **Shape:** 1200 rows × 13 feature columns (10 numerical, 3 categorical)
- **Missingness:** 5.47% overall
- **Class imbalance ratio:** 5.35 (imbalanced=True)

## 2. Exploratory Data Analysis

- Dataset has 1200 rows and 13 feature columns (10 numerical, 3 categorical).
- Overall missingness is 5.47%.
- High-cardinality categorical column(s) detected: ['category_id_highcard'] — candidates for target/frequency encoding rather than one-hot.
- Strongest numerical correlation is between 'num_feature_4' and 'num_feature_8' (r=-0.80).
- **Target balance:** Target is imbalanced (majority:minority ratio = 5.35).
- **Correlation note:** Top correlated numerical pairs: [('num_feature_4', 'num_feature_8', -0.8), ('num_feature_2', 'num_feature_8', -0.57), ('num_feature_0', 'num_feature_2', -0.57)].

## 3. Feature Engineering

- **Imputation:** median
- **Encoding:** mixed
- **Scaling:** True
- **PCA:** False
- **Feature selection k:** n/a
- **Imbalance handling:** True

**Reasoning:** Planner decision supplied — used as the primary execution plan; metadata heuristics applied only where the Planner was silent. 10 numerical / 3 categorical columns detected. Imputation='median' chosen based on the Planner recommendation. Encoding='mixed': low-cardinality categoricals (['category_region', 'category_segment']) get one-hot, high-cardinality columns (['category_id_highcard']) get ordinal encoding to avoid dimensionality blow-up. Scaling=enabled (per Planner). Imbalance handling enabled (class_weight='balanced' where supported) (per Planner).

## 4. Model Selection

**Chosen model:** `LightGBM`

**Candidates considered, in ranked order:**

1. **CatBoost** — (score=4.5) Gradient/ensemble tree model — robust to feature scale and nonlinear interactions. Tolerates the high-cardinality categorical column(s) better than one-hot-heavy linear models. Supports native class-weighting to counter the observed target imbalance. Best native categorical-feature handling of the ensemble family.
2. **XGBoost** — (score=4.0) Gradient/ensemble tree model — robust to feature scale and nonlinear interactions. Tolerates the high-cardinality categorical column(s) better than one-hot-heavy linear models. Supports native class-weighting to counter the observed target imbalance.
3. **LightGBM** — (score=4.0) Gradient/ensemble tree model — robust to feature scale and nonlinear interactions. Tolerates the high-cardinality categorical column(s) better than one-hot-heavy linear models. Supports native class-weighting to counter the observed target imbalance.
4. **Random Forest** — (score=3.5) Gradient/ensemble tree model — robust to feature scale and nonlinear interactions. Tolerates the high-cardinality categorical column(s) better than one-hot-heavy linear models.
5. **Logistic Regression** — (score=2.0) Simple, interpretable, fast-to-train baseline; well suited to a low-dimensional (13 feature) setting where linear decision boundaries are a reasonable prior.
6. **SVM** — (score=1.7) Kernel methods can capture nonlinear boundaries on small-to-medium data (1200 rows), but training cost grows poorly beyond a few thousand rows.

## 5. Metrics

Primary scorer: `f1_weighted`

**Best model held-out metrics:**

| Metric | Value |
|---|---|
| accuracy | 0.9375 |
| precision | 0.9369 |
| recall | 0.9375 |
| f1 | 0.9330 |
| roc_auc | 0.8868 |

**All candidates compared:**

| Model | ACCURACY | PRECISION | RECALL | F1 | ROC_AUC |
|---|---|---|---|---|---|
| CatBoost | 0.9167 | 0.9127 | 0.9167 | 0.9101 | 0.8433 |
| XGBoost | 0.9083 | 0.9025 | 0.9083 | 0.9025 | 0.8659 |
| **LightGBM** | 0.9375 | 0.9369 | 0.9375 | 0.9330 | 0.8868 |

## 6. Explainability (SHAP)

Explainer: `TreeExplainer`

**Top 5 features by mean |SHAP|:**

1. `num__num_feature_6` — 2.2229
2. `num__num_feature_7` — 2.1330
3. `num__num_feature_2` — 2.0123
4. `num__num_feature_0` — 1.9454
5. `num__num_feature_9` — 1.4495

- Top driver of `LightGBM` predictions: **num__num_feature_6** (mean |SHAP| = 2.2229).
- Explained 200 of 240 held-out rows using TreeExplainer against a 100-row background sample.

**Critic-facing explainability notes:**

- 2 domain feature(s) contribute negligible signal and may be candidates for removal: ['category_region', 'category_segment'].

## 7. Critic Review

**Recommendation:** `APPROVE`

- Overfitting detected: False
- Leakage suspected: False
- Feature engineering OK: True
- Metrics acceptable: True

**Comments:** Reviewed 'LightGBM': train f1=1.000 vs held-out f1=0.933 (gap=0.067). No material issues found; approving the pipeline.

## 8. Memory

**Similar past runs retrieved before planning:**
  - run 1 (similarity=1.00, usefulness=0.96, model=LightGBM)

## 9. Future Improvements

- Address class imbalance explicitly (e.g. class weighting, SMOTE, or a threshold-tuned decision boundary) rather than relying on weighted averaging in the metrics alone.
- Investigate the missingness mechanism (MCAR/MAR/MNAR) rather than defaulting to median/most-frequent imputation, since >5% of cells are missing overall.
- Revisit encoding for high-cardinality columns (category_id_highcard) — target or frequency encoding may generalize better than the current strategy.
- Expand the Optuna budget past 20 trials on `LightGBM` specifically, now that it is the identified best candidate rather than one of several unknowns.

## Experience & Memory
- **Experience score:** 0.8116
- **Generalization score:** 0.75
- **Confidence:** 0.7
- **Scorer reasoning:** performance=0.93, robustness=0.50, overfitting_penalty=1.00, critic_confidence=0.90, planner_confidence=0.50, preprocessing_quality=1.00 => experience_score=0.812 (retain)

- **Memory update action:** merge
- **Stored?** Yes
- **Replaced existing memory?** No
- **Reason:** Comparable quality (0.787 vs 0.787) — merged in place.

**Retrieved similar memories considered:**
- run 1 (similarity=1.00, model=LightGBM)

## Appendix: Execution Trace

Total revision iterations: 1

1. `dataset_analyzer` — status=ok
2. `memory_retrieval` — status=ok
3. `planner` — status=ok
4. `eda_agent` — status=ok
5. `feature_engineering_agent` — status=ok
6. `model_recommendation_agent` — status=ok
7. `training_agent` — status=ok
8. `shap_explainability` — status=ok
9. `critic_agent` — status=ok
10. `experience_scorer` — status=ok
11. `memory_update_policy` — status=ok
