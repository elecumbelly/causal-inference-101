---
title: "Lesson 21: External Validity and Transportability"
subtitle: "From First Principles to Modern Causal Inference"
---

# Lesson 21: External Validity and Transportability

## Opening Story: From Lab to Real World

A drug works in a clinical trial with carefully selected patients. But will it work in the real world with diverse patients, different doctors, and varying healthcare systems? This is the question of external validity.

Transportability theory asks: when can we take causal knowledge from one setting and apply it to another?

---

## Learning Objectives

By the end of this lesson, you should be able to:

1. Define external validity and transportability
2. Explain when causal effects can be transported
3. Implement selection diagrams
4. Apply transportability methods
5. Recognize limitations of external validity

---

```{figure} ../figures/instructional/transportability.svg
:name: lesson-21-transportability
:alt: Transporting an effect requires modelling population differences and invariant mechanisms.
:width: 100%

Transporting an effect requires modelling population differences and invariant mechanisms.
```

---

## 21.1 External Validity

### Definition

External validity is the extent to which causal effects estimated in one context can be generalized to other contexts.

### Threats to External Validity

1. **Population differences**: Different demographics
2. **Treatment variation**: Different implementations
3. **Outcome measurement**: Different definitions
4. **Contextual effects**: Different environments

---

## 21.2 Transportability Theory

### Selection Diagrams

A selection diagram is a DAG with additional nodes (S) representing differences between source and target populations.

```
S → X
↓
Y ← T
```

### When Effects Can Be Transported

Effects can be transported if:
1. The treatment mechanism is the same
2. The outcome mechanism is the same
3. Differences are captured by selection nodes

---

## 21.3 Implementation

```python
import numpy as np
import pandas as pd

np.random.seed(42)

# Source population
n_source = 1000
X_source = np.random.normal(0, 1, n_source)
T_source = np.random.binomial(1, 0.5, n_source)
Y_source = 2 * T_source + 1.5 * X_source + np.random.normal(0, 1, n_source)

# Target population (different distribution of X)
n_target = 500
X_target = np.random.normal(1, 1.5, n_target)  # Different mean and variance
T_target = np.random.binomial(1, 0.5, n_target)

# Estimate treatment effect in source
from sklearn.linear_model import LinearRegression
model_source = LinearRegression()
model_source.fit(np.column_stack([T_source, X_source]), Y_source)
ate_source = model_source.coef_[0]

# Transport effect to target
# If treatment effect is constant across X, same effect applies
print(f"ATE in source: {ate_source:.3f}")
print(f"ATE in target (same if effect is constant): {ate_source:.3f}")

# If effect varies with X, need to adjust
# Simulate heterogeneous effect
Y_source_het = 2 * T_source + (1.5 + 0.5 * X_source) * T_source + np.random.normal(0, 1, n_source)

model_het = LinearRegression()
model_het.fit(np.column_stack([T_source, X_source]), Y_source_het)

# Predicted effect in target
X_target_mean = X_target.mean()
effect_target = model_het.coef_[0] + model_het.coef_[1] * X_target_mean
print(f"\nHeterogeneous effect in source: {model_het.coef_[0]:.3f}")
print(f"Predicted effect in target: {effect_target:.3f}")
```

---

## 21.4 Common Mistakes

1. **Assuming homogeneity**: Effects often vary across populations
2. **Ignoring selection**: Who is in the study matters
3. **Overgeneralizing**: Results are context-specific
4. **Not validating**: Test external validity when possible

---

## 21.5 Knowledge Check

### Multiple Choice

1. **External validity asks:**
   A) Is the study internally valid?
   B) Can results be generalized?
   C) Is the sample representative?
   D) Are the methods appropriate?

2. **Transportability requires:**
   A) Identical populations
   B) Known differences between populations
   C) No differences
   D) Random sampling

3. **Selection diagrams:**
   A) Show causal structure
   B) Show differences between populations
   C) Both A and B
   D) Neither

4. **Treatment effect heterogeneity:**
   A) Helps transportability
   B) Complicates transportability
   C) Has no effect
   D) Is always present

5. **External validity is:**
   A) Always guaranteed
   B) Never important
   C) Context-dependent
   D) Only for experiments

### Short Answer

6. **Explain why a treatment effect from one study may not apply to another population.**

7. **What is a selection diagram and how is it used?**

8. **How does treatment effect heterogeneity affect transportability?**

9. **When can we transport causal effects between populations?**

10. **Give an example where external validity might be violated.**

---

## Watch and Connect

```{admonition} Recommended lecture
:class: tip

[**Transfer learning and transportability**](https://www.youtube.com/watch?v=JNq4oCV9C5k&list=PLoazKTcS0Rzb6bb9L508cyJ1z-U9iWkA0&index=77)

Treats external validity as a structured comparison between source and target populations.

**Active-viewing prompt:** Identify which mechanisms are assumed invariant across populations.
```

---

## 21.6 Summary

1. **External validity** concerns generalization of results
2. **Transportability** asks when effects can be moved across contexts
3. **Selection diagrams** formalize differences between populations
4. **Treatment effect heterogeneity** complicates transportability
5. **Always consider** the target population

---

## 21.7 Further Reading

- Pearl, J. & Bareinboim, E. (2014). "External Validity: From Do-Calculus to Transportability." *AAAI Workshop on Causal Reasoning*.
- Dahabreh, I.J. et al. (2019). "External Validity in Randomized Controlled Trials." *Epidemiologic Reviews*.


---

## Worked Examples

### Example 1: Medicine — Trial to Clinical Practice

A drug works in a randomized trial with strict inclusion criteria. But in clinical practice, patients are older, sicker, and have more comorbidities. The treatment effect may not transport to this broader population.

### Example 2: Economics — Training Program Generalizability

A job training program shows positive effects in urban areas. Will it work in rural areas? Differences in labor markets, demographics, and program implementation may limit transportability.

### Example 3: Technology — Platform A/B Test to Global Rollout

A feature tested on US users may not work the same way in other cultures. External validity requires understanding how context affects treatment effects.

### Example 4: Policy — Pilot Program to National Scale

A pilot program shows positive results in one city. Scaling nationally introduces new challenges: different administrative capacities, population compositions, and implementation quality.

---

## Diagnostics: Assessing External Validity

### Transportability Conditions

For causal effects to transport from study population to target population:
1. **Same causal structure**: The DAG must be the same in both populations
2. **No effect modification by population**: The CATE function must be the same
3. **Representativeness**: The study sample must cover the range of the target population

```python
import numpy as np
import pandas as pd

def check_transportability(df_study, df_target, treatment, outcome, covariates):
    """Check if causal effects can be transported."""

    # Check covariate overlap
    print("Covariate Distribution Comparison:")
    print("=" * 50)

    for cov in covariates:
        study_mean = df_study[cov].mean()
        target_mean = df_target[cov].mean()
        diff = abs(study_mean - target_mean) / df_study[cov].std()

        status = "OK" if diff < 0.5 else "WARN"
        print(f"  {cov}: Study={study_mean:.2f}, Target={target_mean:.2f}, SMD={diff:.2f} [{status}]")

    # Check outcome model stability
    from sklearn.linear_model import LinearRegression

    model_study = LinearRegression().fit(df_study[covariates], df_study[outcome])
    model_target = LinearRegression().fit(df_target[covariates], df_target[outcome])

    r_sq_study = model_study.score(df_study[covariates], df_study[outcome])
    r_sq_target = model_target.score(df_target[covariates], df_target[outcome])

    print(f"\nOutcome model R-squared:")
    print(f"  Study: {r_sq_study:.3f}")
    print(f"  Target: {r_sq_target:.3f}")
```

### Sensitivity Analysis for Transportability

```python
def transport_sensitivity(ate_study, sensitivity_params):
    """Assess sensitivity of transported effect to violations."""

    results = {}

    for param_name, param_value in sensitivity_params.items():
        transported_effect = ate_study * param_value
        results[param_name] = transported_effect
        print(f"  {param_name}: {transported_effect:.3f}")

    return results
```

---

## Interpretation Workshop

### When External Validity Fails

- **Different populations**: Trial participants differ from target population
- **Different contexts**: Treatment delivery differs between study and practice
- **Different time periods**: Effects may change over time
- **Hawthorne effects**: Study participation itself affects behavior

### Reporting External Validity

When presenting results:
1. Clearly describe the study population
2. Compare to the target population on observables
3. Discuss potential effect modifiers
4. Present sensitivity analyses for transportability

---

## Practical Application

### Improving External Validity

1. **Inclusive trial design**: Broaden inclusion criteria
2. **Pragmatic trials**: Conduct trials in real-world settings
3. **Stratified analysis**: Report effects by subgroup
4. **Replication**: Conduct the study in multiple settings
5. **Meta-analysis**: Combine results across diverse studies

### Weighting for Transportability

```python
def transport_weights(df_study, df_target, covariates):
    """Compute weights to transport study to target population."""

    from sklearn.linear_model import LogisticRegression

    # Combine datasets
    df_study = df_study.copy()
    df_study['population'] = 1  # Study

    df_target = df_target.copy()
    df_target['population'] = 0  # Target

    df_combined = pd.concat([df_study, df_target])

    # Fit propensity model
    model = LogisticRegression(max_iter=1000)
    model.fit(df_combined[covariates], df_combined['population'])

    # Get weights for study population
    probs = model.predict_proba(df_study[covariates])[:, 1]
    weights = (1 - probs) / probs  # Target vs study odds

    return weights / np.mean(weights)  # Normalize
```

---

## Limitations

- Cannot test transportability assumptions empirically
- Effect modification may be unmeasured
- Weights can be unstable with poor covariate overlap
- Context-specific factors may be unobservable

---

## Exercises

1. **Transportability check**: Compare a clinical trial population to a real-world population. What are the key differences?
2. **Weighting**: Implement transport weighting. How do the results change?
3. **Sensitivity analysis**: How much effect modification would need to exist to invalidate transport?
4. **Critique**: Find a study that claims generalizability. Evaluate the evidence.

---

## Projects

### Project 1: Transportability Analysis
Conduct a transportability analysis moving causal findings from a trial to a real-world population.

### Project 2: External Validity Assessment
Assess the external validity of a published RCT by comparing the study population to the target population.


---


---

## Companion Practice

Move from reading to doing with the complete course materials:

- [Download the worked notebook](../notebooks/21-external-validity.ipynb)
- [Download the practice lab](../labs/lab21-external-validity-practice.ipynb)
- [Download the lab solution](../solutions/lab21-solutions.ipynb) — attempt the lab before opening this

The worked notebook demonstrates the lesson's core estimator. The practice lab asks you to make the identification and diagnostic choices yourself.
