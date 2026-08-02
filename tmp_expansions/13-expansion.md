

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
