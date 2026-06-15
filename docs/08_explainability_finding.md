# Explainability: what drives the detector, and is it robust

## Method

SHAP (TreeExplainer) was run on held-out predictions to find which features
drive the detector, both globally and for a single alarm. Source:
notebooks/19_shap_explain.py.

## Finding: one feature stood out, so it was investigated

SHAP showed QGL drift from baseline (gas-lift flow drifting from the well's
early baseline) dominating global importance by a wide margin, with
temperature and pressure features contributing less. When a single feature
dominates this hard, the responsible check is whether the model is resting
on it, possibly exploiting a dataset artifact that would not generalise.

## Ablation test

The detector was retrained with QGL removed, and with QGL only. Source:
notebooks/20_qgl_ablation.py.

| Feature set | Mean AP | Std across wells |
|---|---|---|
| All features | 0.975 | 0.016 |
| Without QGL | 0.886 | 0.113 |
| QGL only | 0.920 | 0.099 |

## Conclusion

The result is nuanced and both parts matter:

- The model does not collapse without QGL (0.975 to 0.886), so it has
  genuine signal distributed across temperature, pressure, and choke
  features. It is not a one-feature artifact.
- QGL alone reaches 0.920, so gas-lift flow drift is the single strongest
  physical predictor. This is consistent with the physics: hydrate
  formation restricts flow, and gas-lift flow drifting from baseline is a
  sensible early indicator of that restriction.
- QGL is also stabilising: without it, fold-to-fold variance rises sharply
  (std 0.016 to 0.113), so it helps the model generalise consistently across
  wells.

The honest statement is a detector with robust, distributed signal whose
single strongest predictor (QGL drift) reflects real flow-restriction
physics, verified by ablation rather than assumed from a SHAP plot.