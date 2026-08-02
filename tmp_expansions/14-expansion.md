

---

## Worked Examples

### Example 1: Medicine — Drug Effect by Age Group

A drug may work better for younger patients than older ones. CATE estimation identifies which patients benefit most, enabling personalized treatment recommendations.

### Example 2: Economics — Job Training by Education Level

A job training program may have larger effects for high school graduates than for college graduates. Policy targeting depends on this heterogeneity — resources should go where they have the largest impact.

### Example 3: Technology — Feature Effect by User Segment

A new feature may increase engagement for power users but not casual users. Personalized rollout depends on CATE estimation to avoid wasting resources on non-responsive segments.

### Example 4: Policy — Education Intervention by Socioeconomic Status

A tutoring program may have larger effects for low-SES students. Equity considerations require understanding effect heterogeneity to ensure fair resource allocation.

---

## Diagnostics: CATE Estimation

### Meta-Learners

```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier

class XLearner:
    """X-Learner for CATE estimation."""

    def __init__(self):
        self.models = {}

    def fit(self, X, T, Y):
        """Fit X-Learner."""
        treated = T == 1
        control = T == 0

        # Stage 1: Outcome models for treatment and control
        self.models['outcome_treated'] = GradientBoostingRegressor()
        self.models['outcome_control'] = GradientBoostingRegressor()

        self.models['outcome_treated'].fit(X[treated], Y[treated])
        self.models['outcome_control'].fit(X[control], Y[control])

        # Stage 2: Impute treatment effects
        tau_treated = Y[treated] - self.models['outcome_control'].predict(X[treated])
        tau_control = self.models['outcome_treated'].predict(X[control]) - Y[control]

        # Stage 3: Propensity score model
        self.models['propensity'] = RandomForestClassifier()
        self.models['propensity'].fit(X, T)

        # Stage 4: CATE models
        self.models['cate_treated'] = GradientBoostingRegressor()
        self.models['cate_control'] = GradientBoostingRegressor()

        self.models['cate_treated'].fit(X[treated], tau_treated)
        self.models['cate_control'].fit(X[control], tau_control)

        return self

    def predict(self, X):
        """Predict CATE."""
        e = self.models['propensity'].predict_proba(X)[:, 1]
        tau_treated = self.models['cate_treated'].predict(X)
        tau_control = self.models['cate_control'].predict(X)

        tau = e * tau_control + (1 - e) * tau_treated

        return tau
```

### Validation of CATE Estimates

```python
def validate_cate(X, T, Y, cate_estimates, n_groups=5):
    """Validate CATE estimates by comparing within groups."""

    sorted_idx = np.argsort(cate_estimates)
    groups = np.array_split(sorted_idx, n_groups)

    print("CATE Validation by Predicted Effect Group:")
    print("=" * 50)

    for i, group in enumerate(groups):
        group_tau = cate_estimates[group]
        mean_tau = np.mean(group_tau)

        treated = T[group] == 1
        control = T[group] == 0

        if treated.sum() > 0 and control.sum() > 0:
            y_treated = Y[group][treated].mean()
            y_control = Y[group][control].mean()
            observed_effect = y_treated - y_control

            print(f"Group {i+1}: Predicted CATE = {mean_tau:.3f}, Observed effect = {observed_effect:.3f}")
```

---

## Interpretation Workshop

### When Heterogeneity Matters

- **Policy targeting**: Who should receive the intervention?
- **Equity analysis**: Does the intervention benefit disadvantaged groups?
- **Personalization**: Can we tailor treatment to individual characteristics?
- **External validity**: Will the effect generalize to new populations?

### Interpreting CATE Plots

- Positive CATE: Subgroup benefits from treatment
- Negative CATE: Subgroup is harmed by treatment
- Zero CATE: No effect for this subgroup
- Wide confidence intervals: Uncertain about effect in this subgroup

---

## Practical Application

### Decision Trees for CATE

```python
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt

def cate_tree(X, T, Y, feature_names=None):
    """Use decision tree to identify subgroups with different effects."""

    treated = T == 1
    control = T == 0

    model_treated = GradientBoostingRegressor()
    model_control = GradientBoostingRegressor()

    model_treated.fit(X[treated], Y[treated])
    model_control.fit(X[control], Y[control])

    tau_hat = model_treated.predict(X) - model_control.predict(X)

    tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=50)
    tree.fit(X, tau_hat)

    plt.figure(figsize=(15, 8))
    plot_tree(tree, feature_names=feature_names, filled=True, fontsize=10, proportion=True)
    plt.title("Subgroups with Different Treatment Effects")
    plt.tight_layout()

    return tree, tau_hat
```

---

## Limitations

- Sample size: CATE estimation requires large samples
- Overfitting: Easy to find spurious heterogeneity
- Multiple testing: Many subgroup comparisons inflate false positives
- External validity: CATE estimates may not generalize

---

## Exercises

1. **Implement CATE estimation**: Use the X-Learner to estimate heterogeneous treatment effects. How do results compare to a simple T-Learner?
2. **Subgroup analysis**: Use decision trees to identify subgroups. Are the subgroups interpretable?
3. **Validation**: How would you validate CATE estimates out-of-sample?
4. **Policy implication**: If you found that the treatment works better for men than women, what policy implications would this have?

---

## Projects

### Project 1: CATE Comparison Study
Compare T-Learner, S-Learner, X-Learner, and R-Learner on simulated data with known heterogeneity.

### Project 2: HTE in Practice
Apply CATE estimation to a real dataset. Identify which subgroups benefit most and least from the treatment.
