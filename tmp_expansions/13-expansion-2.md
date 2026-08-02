

---

## Additional Advanced Content

### Potential Outcomes Framework for Mediation

The causal mediation literature distinguishes between:

1. **Natural Direct Effect (NDE)**: Effect of treatment on outcome when mediator is held at its natural level under control
2. **Natural Indirect Effect (NIE)**: Effect of treatment operating through the mediator
3. **Total Effect**: NDE + NIE (in linear models without interaction)

```python
import numpy as np
import pandas as pd

def natural_effects_estimation(df, treatment, mediator, outcome, covariates=None):
    """Estimate natural direct and indirect effects."""

    # Step 1: Outcome model Y ~ T + M + covariates
    import statsmodels.api as sm

    X_out = df[[treatment, mediator] + (covariates or [])]
    X_out = sm.add_constant(X_out)
    model_out = sm.OLS(df[outcome], X_out).fit()

    # Step 2: Mediator model M ~ T + covariates
    X_med = df[[treatment] + (covariates or [])]
    X_med = sm.add_constant(X_med)
    model_med = sm.OLS(df[mediator], X_med).fit()

    # Step 3: Predict potential outcomes
    n = len(df)

    # Y under T=1, M=M(1) - natural state
    df_t1 = df.copy()
    df_t1[treatment] = 1
    Y_t1_m1 = model_out.predict(sm.add_constant(df_t1[[treatment, mediator] + (covariates or [])]))

    # Y under T=0, M=M(0) - natural state
    df_t0 = df.copy()
    df_t0[treatment] = 0
    Y_t0_m0 = model_out.predict(sm.add_constant(df_t0[[treatment, mediator] + (covariates or [])]))

    # Y under T=1, M=M(0) - cross-world
    M_t0 = model_med.predict(sm.add_constant(df_t0[[treatment] + (covariates or [])]))
    df_cross = df.copy()
    df_cross[treatment] = 1
    df_cross[mediator] = M_t0
    Y_t1_m0 = model_out.predict(sm.add_constant(df_cross[[treatment, mediator] + (covariates or [])]))

    # Natural Direct Effect
    nde = np.mean(Y_t1_m0 - Y_t0_m0)

    # Natural Indirect Effect
    nie = np.mean(Y_t1_m1 - Y_t1_m0)

    # Total Effect
    te = np.mean(Y_t1_m1 - Y_t0_m0)

    print("Natural Effects Estimation:")
    print("=" * 50)
    print(f"Natural Direct Effect (NDE): {nde:.4f}")
    print(f"Natural Indirect Effect (NIE): {nie:.4f}")
    print(f"Total Effect (TE): {te:.4f}")
    print(f"NIE/TE (Proportion mediated): {nie/te:.1%}" if te != 0 else "N/A")

    return {'nde': nde, 'nie': nie, 'te': te}
```

### Sensitivity Analysis for Mediation

```python
def mediation_sensitivity_rho(a, b, rho_range=np.arange(0, 0.8, 0.1)):
    """Sensitivity analysis for mediation under unmeasured confounding."""

    results = []

    for rho in rho_range:
        # Under confounding bias parameter rho
        # True indirect effect = a*b - rho*se_a*se_b (simplified)

        # Sensitivity parameter: correlation between errors of mediator and outcome
        bias = rho * np.std(a) * np.std(b)  # Simplified
        adjusted_indirect = a * b - bias

        results.append({
            'rho': rho,
            'bias': bias,
            'adjusted_indirect': adjusted_indirect
        })

    results_df = pd.DataFrame(results)

    print("Mediation Sensitivity Analysis:")
    print("=" * 50)
    print(results_df.to_string(index=False))

    # Find rho where indirect effect becomes zero
    zero_rho = results_df[results_df['adjusted_indirect'] <= 0]['rho'].min()
    if not np.isnan(zero_rho):
        print(f"\nIndirect effect becomes zero at rho = {zero_rho:.2f}")

    return results_df
```

### When Mediation Analysis Fails

Common pitfalls:
1. **Reciprocal causation**: M affects T and T affects M
2. **Treatment-mediator interaction**: Effect of M depends on T
3. **Time-varying confounding**: Confounders affected by prior treatment
4. **Mediator-outcome confounding**: U affects both M and Y

```python
def check_mediation_assumptions(df, treatment, mediator, outcome):
    """Check key mediation assumptions."""

    print("Mediation Assumption Checks:")
    print("=" * 50)

    # 1. Temporal ordering
    if 'time' in df.columns:
        t_time = df.groupby(treatment)['time'].mean()
        print(f"1. Temporal ordering: Treatment before mediator? ", end="")
        if len(t_time) == 2:
            print("CHECK MANUALLY")
        else:
            print("NO TIME VARIABLE")

    # 2. No treatment-mediator interaction
    import statsmodels.api as sm
    df['TM_interaction'] = df[treatment] * df[mediator]
    model = sm.OLS(df[outcome], sm.add_constant(df[[treatment, mediator, 'TM_interaction']])).fit()
    interaction_pval = model.pvalues['TM_interaction']
    print(f"2. Treatment-mediator interaction p-value: {interaction_pval:.4f}")
    print(f"   {'WARN: Interaction may be present' if interaction_pval < 0.1 else 'OK: No strong interaction'}")

    # 3. Parallel trends (for longitudinal data)
    print("3. Parallel trends: CHECK VISUALLY with pre-treatment periods")

    return {'interaction_pval': interaction_pval}
```
