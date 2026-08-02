

---

## Worked Examples

### Example 1: Economics — California Tobacco Control (Abadie et al. 2010)

California implemented Proposition 99 (tobacco tax) in 1988. The synthetic control is a weighted combination of other states that best matches California's pre-treatment cigarette consumption trends.

### Example 2: Policy — German Reunification (Abadie et al. 2015)

West Germany reunified with East Germany in 1990. The synthetic control is a weighted combination of OECD countries that best matches West Germany's pre-reunification GDP trends.

### Example 3: Technology — EU GDPR

The EU implemented GDPR in 2018. A synthetic EU is constructed from non-EU countries to estimate GDPR's effect on tech innovation.

### Example 4: Health — Oregon Health Insurance Experiment

Oregon expanded Medicaid via lottery. The synthetic control method constructs a comparison for lottery winners using non-winners.

---

## Diagnostics: Synthetic Control Quality

### Pre-Treatment Fit Assessment

```python
import numpy as np

def assess_synthetic_control(pre_treatment_actual, pre_treatment_synthetic):
    """Assess quality of synthetic control match."""

    rmspe = np.sqrt(np.mean((pre_treatment_actual - pre_treatment_synthetic) ** 2))
    mape = np.mean(np.abs(pre_treatment_actual - pre_treatment_synthetic) /
                   np.abs(pre_treatment_actual)) * 100
    corr = np.corrcoef(pre_treatment_actual, pre_treatment_synthetic)[0, 1]

    print(f"Pre-Treatment Fit Quality:")
    print(f"  RMSPE: {rmspe:.3f}")
    print(f"  MAPE: {mape:.1f}%")
    print(f"  Correlation: {corr:.3f}")

    return rmspe, mape, corr
```

### Permutation Tests

```python
def permutation_test_sc(treated_outcome, synthetic_outcome, donor_outcomes):
    """Permutation test for synthetic control significance."""

    gap_treated = treated_outcome - synthetic_outcome

    placebo_gaps = []
    for donor in donor_outcomes:
        placebo_gaps.append(donor['actual'] - donor['synthetic'])

    all_gaps = np.array([gap_treated] + placebo_gaps)
    p_value = np.mean(np.abs(all_gaps[1:]) >= np.abs(gap_treated))

    print(f"Permutation test p-value: {p_value:.3f}")

    return p_value
```

---

## Interpretation Workshop

### Reading Synthetic Control Studies

Key questions:
1. **How was the synthetic control constructed?** What variables were used for matching?
2. **How well does it fit pre-treatment outcomes?** (RMSPE, visual inspection)
3. **What donors were included?** Are they comparable?
4. **Is the post-treatment divergence statistically significant?** (Permutation test)
5. **Are there robustness checks?** (Leave-one-out, alternative donors)

### The Role of Donor Pool

The synthetic control is only as good as the donor pool:
- Too few donors: Limited flexibility to match pre-treatment trends
- Too many donors: Risk of overfitting
- Similar donors: Better matches, but may share the treatment

---

## Practical Application

### Implementing Synthetic Control

```python
import numpy as np
from scipy.optimize import minimize

class SyntheticControl:
    """Synthetic control method implementation."""

    def __init__(self, treated_unit, donor_units):
        self.treated_unit = treated_unit
        self.donor_units = donor_units
        self.weights = None

    def fit(self, treated_pre, donor_pre):
        """Find optimal weights for synthetic control."""

        n_donors = len(self.donor_units)

        def objective(weights):
            synthetic = donor_pre @ weights
            return np.sqrt(np.mean((treated_pre - synthetic) ** 2))

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0, 1) for _ in range(n_donors)]
        w0 = np.ones(n_donors) / n_donors

        result = minimize(objective, w0, method='SLSQP',
                         bounds=bounds, constraints=constraints)

        self.weights = result.x

        print("Synthetic Control Weights:")
        for unit, weight in zip(self.donor_units, self.weights):
            if weight > 0.01:
                print(f"  {unit}: {weight:.3f}")

        return self

    def predict(self, donor_data):
        """Predict synthetic control."""
        return donor_data @ self.weights

    def estimate_effect(self, treated_post, synthetic_post):
        """Estimate treatment effect."""
        effect = np.mean(treated_post - synthetic_post)
        print(f"Treatment Effect: {effect:.3f}")
        return effect
```

---

## Limitations

- **Single treated unit**: Method designed for N=1 case studies
- **Pre-treatment fit**: No guarantee of good pre-treatment match
- **Extrapolation**: Synthetic control may not represent counterfactual
- **Inference**: Permutation tests may have low power with few donors

---

## Exercises

1. **Implement synthetic control**: Use the code to construct a synthetic control for a treated unit. How well does it match pre-treatment outcomes?

2. **Donor pool sensitivity**: Remove one donor at a time and re-estimate. How sensitive are results to the donor pool?

3. **Covariate selection**: What happens when you include different covariates in the matching? How do weights change?

4. **Inference**: Conduct a permutation test. Is the post-treatment effect significant?

---

## Projects

### Project 1: Synthetic Control Application
Apply synthetic control to a policy intervention of your choice. Select treated unit and donor pool, construct synthetic control, assess pre-treatment fit, estimate treatment effect, and conduct robustness checks.

### Project 2: Augmented Synthetic Control
Implement the augmented synthetic control method (Ben-Michael et al. 2021) that combines synthetic control with outcome modeling.
