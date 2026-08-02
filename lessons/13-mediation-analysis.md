---
title: "Lesson 13: Mediation Analysis"
subtitle: "From First Principles to Modern Causal Inference"
---

# Lesson 13: Mediation Analysis

## Opening Story: Education and Health

Education is associated with better health. But why? Is it because education leads to higher income, which enables better healthcare? Or does education change health behaviors directly?

Mediation analysis tries to answer these questions by decomposing the total effect of an intervention into direct and indirect effects through specific pathways.

---

## Learning Objectives

By the end of this lesson, you should be able to:

1. Define direct and indirect effects
2. Explain the sequential ignorability assumption
3. Implement mediation analysis using regression
4. Conduct sensitivity analysis for mediation
5. Recognize limitations of standard mediation methods

---

```{figure} ../figures/instructional/mediation-effects.svg
:name: lesson-13-mediation-effects
:alt: Direct and indirect effects correspond to different nested counterfactual comparisons.
:width: 100%

Direct and indirect effects correspond to different nested counterfactual comparisons.
```

---

## 13.1 The Mediation Framework

### The Setup

```
       M (Mediator)
      ↗       ↘
X (Treatment) → Y (Outcome)
```

### Natural Direct Effect

Holding the mediator to the value it would naturally take under control:

$$NDE = E[Y(1, M(0)) - Y(0, M(0))]$$

### Natural Indirect Effect

Holding treatment fixed at 1 while changing the mediator from its control to treatment value:

$$NIE = E[Y(1, M(1)) - Y(1, M(0))]$$

### Total Effect

$$TE = E[Y(1,M(1)) - Y(0,M(0))] = NDE + NIE$$

These are cross-world counterfactual quantities. Their identification requires strong assumptions, including no treatment-induced confounder of the mediator–outcome relationship.

---

## 13.2 Sequential Ignorability

The key assumption for identifying causal mediation effects:

1. No unmeasured confounders of treatment → outcome
2. No unmeasured confounders of mediator → outcome
3. No unmeasured confounders of treatment → mediator

This is stronger than the assumption for treatment effects alone.

---

## 13.3 Regression Approach

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm

np.random.seed(42)
n = 1000

# Generate data
X = np.random.binomial(1, 0.5, n)
M = 0.5 + 1.5 * X + np.random.normal(0, 1, n)
Y = 2 + 1 * X + 2 * M + np.random.normal(0, 1, n)

# Step 1: Total effect (Y on X)
model_total = sm.OLS(Y, sm.add_constant(X)).fit()
total_effect = model_total.params[1]

# Step 2: Mediator on X
model_mediator = sm.OLS(M, sm.add_constant(X)).fit()
a_path = model_mediator.params[1]

# Step 3: Y on X and M
model_outcome = sm.OLS(Y, sm.add_constant(np.column_stack([X, M]))).fit()
b_path = model_outcome.params[2]
direct_effect = model_outcome.params[1]

# Indirect effect
indirect_effect = a_path * b_path

print(f"Total effect: {total_effect:.3f}")
print(f"Direct effect: {direct_effect:.3f}")
print(f"Indirect effect: {indirect_effect:.3f}")
print(f"Proportion mediated: {indirect_effect/total_effect:.3f}")
```

---

## 13.4 Common Mistakes

1. **Ignoring unmeasured mediator-outcome confounders**
2. **Assuming linearity without justification**
3. **Not testing sequential ignorability**
4. **Interpreting correlation as causation in mediation**

---

## 13.5 Knowledge Check

### Multiple Choice

1. **The indirect effect measures:**
   A) Effect through the mediator
   B) Effect not through the mediator
   C) Total effect
   D) No effect

2. **Sequential ignorability requires:**
   A) No confounders at all
   B) No unmeasured confounders at specific stages
   C) Perfect randomization
   D) Large sample size

3. **Proportion mediated is:**
   A) Indirect / Total
   B) Direct / Total
   C) Total / Indirect
   D) Direct / Indirect

4. **Mediation analysis is:**
   A) Always causal
   B) Causal under sequential ignorability
   C) Never causal
   D) Only for experiments

5. **A mediator is:**
   A) A confounder
   B) A collider
   C) A variable on the causal pathway
   D) An outcome

### Short Answer

6. **Explain why sequential ignorability is stronger than unconfoundedness.**

7. **What happens if there's an unmeasured confounder between mediator and outcome?**

8. **How can you assess sensitivity to violations of sequential ignorability?**

9. **Give an example where mediation analysis would be useful.**

10. **What are the limitations of standard mediation methods?**

---

## Watch and Connect

```{admonition} Recommended lecture
:class: tip

[**Counterfactuals, mediation, and path-specific effects**](https://www.youtube.com/watch?v=f8PEpthLlN4&list=PLoazKTcS0Rzb6bb9L508cyJ1z-U9iWkA0&index=81)

Shows why decomposing effects requires stronger assumptions than estimating a total effect.

**Active-viewing prompt:** Distinguish a controlled direct effect from a natural direct effect.
```

---

## 13.6 Summary

1. **Mediation analysis** decomposes effects into direct and indirect
2. **Sequential ignorability** is the key assumption
3. **Regression** can implement mediation analysis
4. **Sensitivity analysis** is crucial
5. **Caution** is needed in interpretation

---

## 13.7 Further Reading

- Pearl, J. (2014). "Interpretation and Identification of Causal Mediation." *Psychological Methods*.
- Imai, K., Keele, L., & Tingley, D. (2010). "A General Approach to Causal Mediation Analysis." *Psychological Methods*.


---

## Worked Examples

### Example 1: Medicine — Treatment → Biomarker → Outcome

A drug reduces inflammation (mediator), which improves survival (outcome). Mediation analysis decomposes the total effect into direct (drug → survival) and indirect (drug → inflammation → survival) paths. If the indirect effect is large, the drug works primarily through reducing inflammation.

### Example 2: Economics — Education → Skills → Wages

Education increases cognitive skills (mediator), which increases wages. The indirect effect measures how much of education's return operates through skill accumulation. If the direct effect is also large, education has benefits beyond just building skills (e.g., signaling, networking).

### Example 3: Technology — Design → Usability → Adoption

A new UI design improves usability (mediator), which increases user adoption. The direct effect captures non-usability benefits such as aesthetics or brand perception.

### Example 4: Policy — Training → Confidence → Employment

Job training builds confidence (mediator), which improves employment outcomes. The indirect effect measures the psychological channel, suggesting that programs should also address self-efficacy.

---

## Diagnostics: Mediation Assumptions

### Key Assumptions

1. **No unmeasured confounding** of the mediator-outcome relationship
2. **No unmeasured confounding** of the treatment-mediator relationship
3. **No treatment-mediator interaction** (for standard decomposition)
4. **Temporal ordering**: Mediator affects outcome, not vice versa

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm

def mediation_analysis(df, treatment, mediator, outcome, covariates=None):
    """Complete mediation analysis."""

    # Total effect (Treatment → Outcome)
    X_total = df[[treatment] + (covariates or [])]
    X_total = sm.add_constant(X_total)
    model_total = sm.OLS(df[outcome], X_total).fit()
    total_effect = model_total.params[treatment]

    # Treatment → Mediator
    X_med = df[[treatment] + (covariates or [])]
    X_med = sm.add_constant(X_med)
    model_med = sm.OLS(df[mediator], X_med).fit()
    a_path = model_med.params[treatment]

    # Mediator → Outcome (controlling for treatment)
    X_ind = df[[treatment, mediator] + (covariates or [])]
    X_ind = sm.add_constant(X_ind)
    model_ind = sm.OLS(df[outcome], X_ind).fit()
    b_path = model_ind.params[mediator]
    c_prime = model_ind.params[treatment]

    # Indirect effect (a × b)
    indirect_effect = a_path * b_path

    # Proportion mediated
    proportion_mediated = indirect_effect / total_effect if total_effect != 0 else np.nan

    print("Mediation Analysis Results:")
    print("=" * 50)
    print(f"Total effect (c): {total_effect:.4f}")
    print(f"Direct effect (c'): {c_prime:.4f}")
    print(f"Indirect effect (a×b): {indirect_effect:.4f}")
    print(f"Proportion mediated: {proportion_mediated:.1%}")

    return {
        'total': total_effect,
        'direct': c_prime,
        'indirect': indirect_effect,
        'proportion_mediated': proportion_mediated
    }
```

---

## Interpretation Workshop

### Common Misinterpretations

- "The mediator accounts for X% of the effect" — This assumes no treatment-mediator interaction
- "The direct effect is the effect through other pathways" — This requires strong assumptions
- "Mediation = mechanism" — Mediation is statistical decomposition, not proof of mechanism

### When Mediation Analysis Fails

- Unmeasured confounding of mediator-outcome relationship
- Reciprocal causation (mediator affects treatment and vice versa)
- Treatment-mediator interaction (effect of mediator depends on treatment)

---

## Practical Application

### Sensitivity Analysis for Mediation

```python
def sensitivity_mediation(a, b, se_a, se_b, rho=0.5):
    """Sensitivity analysis for mediation under unmeasured confounding."""

    bias = rho * se_a * se_b
    indirect = a * b
    adjusted_indirect = indirect - bias

    print(f"Indirect effect: {indirect:.4f}")
    print(f"Bias from unmeasured confounding: {bias:.4f}")
    print(f"Adjusted indirect effect: {adjusted_indirect:.4f}")

    return adjusted_indirect
```

---

## Limitations

- Strong causal assumptions required
- Cannot test assumptions empirically
- Sensitivity to unmeasured confounding is high
- Temporal ordering must be clear

---

## Exercises

1. **Implement mediation analysis**: Use the provided code to decompose a total effect into direct and indirect components.
2. **Sensitivity analysis**: How strong would an unmeasured confounder need to be to explain away the indirect effect?
3. **Critique**: Find a mediation study in your field. What assumptions are made? Are they defensible?
4. **Design**: You want to test whether a policy works through a specific mechanism. What data would you need?

---

## Projects

### Project 1: Causal Mediation Analysis
Implement Imai et al. (2010) causal mediation analysis with sensitivity analysis.

### Project 2: Longitudinal Mediation
Implement cross-lagged panel models for longitudinal mediation analysis.


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


---


---

## Companion Practice

Move from reading to doing with the complete course materials:

- [Download the worked notebook](../notebooks/13-mediation-analysis.ipynb)
- [Download the practice lab](../labs/lab13-mediation-analysis-practice.ipynb)
- [Download the lab solution](../solutions/lab13-solutions.ipynb) — attempt the lab before opening this

The worked notebook demonstrates the lesson's core estimator. The practice lab asks you to make the identification and diagnostic choices yourself.
