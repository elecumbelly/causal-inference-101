

---

## Worked Examples

### Example 1: Medicine — Berkson's Bias

A hospital-based study finds that smokers have lower rates of a certain disease than non-smokers. This seems counterintuitive. The explanation is Berkson's bias: smokers who get sick are more likely to be hospitalized (for respiratory issues), but once in the hospital, they are compared to non-smokers who are also hospitalized (for other reasons). Conditioning on being in the hospital introduces collider bias because hospitalization is caused by both smoking (through respiratory disease) and the disease of interest.

### Example 2: Economics — Education and Ability

Among Harvard graduates, education and earnings may appear uncorrelated. This is selection bias: only high-ability individuals get into Harvard. Conditioning on Harvard attendance (a collider of ability and education) distorts the relationship. In the general population, education and earnings are positively correlated.

### Example 3: Technology — Survivorship Bias

A tech company analyzes only successful products to find common success factors. They discover that all successful products had aggressive marketing. But failed products with aggressive marketing are excluded from the analysis. The company concludes that aggressive marketing causes success, when in reality it may have no effect or even be harmful.

### Example 4: Policy — The Healthy User Bias

People who take vitamins appear healthier in observational studies. But health-conscious people are more likely to take vitamins AND more likely to exercise, eat well, and have regular checkups. Conditioning on vitamin use correlates with these unobserved health behaviors, creating the illusion that vitamins improve health.

---

## Diagnostics: Detecting Selection Bias

### Drawing Selection Diagrams

A selection diagram extends a DAG by adding a node S representing the selection mechanism. The key question is: what causes S?

- If S is a descendant of the exposure: Conditioning biases the effect estimate downward
- If S is a descendant of the outcome: Conditioning biases the effect estimate upward
- If S is a common cause of exposure and outcome: Conditioning creates confounding
- If S is independent of both: Conditioning is safe

### Testing for Collider Bias

```python
import numpy as np

def demonstrate_collider_bias(n=10000):
    """Show how conditioning on a collider creates spurious association."""
    U1 = np.random.normal(size=n)  # Independent cause of A
    U2 = np.random.normal(size=n)  # Independent cause of B

    A = U1 + np.random.normal(0, 0.5, size=n)
    B = U2 + np.random.normal(0, 0.5, size=n)

    # A and B are independent overall
    overall_corr = np.corrcoef(A, B)[0,1]
    print(f"Overall correlation between A and B: {overall_corr:.3f}")

    # But condition on S = A + B > 0 (the collider)
    S = (A + B) > 0
    conditional_corr = np.corrcoef(A[S], B[S])[0,1]
    print(f"Conditional correlation (given S>0): {conditional_corr:.3f}")
    print(" Conditioning on a collider creates spurious association!")
```

---

## Interpretation Workshop

### Reading Studies with Potential Selection Bias

When reading any study, ask these questions:

1. **What population is being studied?** Is it representative, or a selected subset?
2. **What selection criteria were used?** Could they introduce bias?
3. **Is the analysis conditional on a collider?** Draw the DAG to check.
4. **Would the results differ in the full population?** How?

### The Table Two Fallacy

Many papers present Table 1 (baseline characteristics by treatment group) and then regress the outcome on all those characteristics plus the treatment. This is problematic if any baseline characteristic is affected by the treatment (a post-treatment variable), because conditioning on it introduces collider bias.

**Correct approach**: Pre-specify covariates based on DAG reasoning, not data-driven selection.

---

## Practical Application

### Avoiding Selection Bias

1. **Define your target population** before analyzing data
2. **Draw the DAG** to identify potential colliders
3. **Avoid conditioning on post-treatment variables**
4. **Use inverse probability weighting** to correct for selection when possible
5. **Report selection criteria** transparently

### Inverse Probability Weighting

When selection is non-random, you can weight observations by the inverse of their probability of being selected:

```python
from sklearn.linear_model import LogisticRegression
import numpy as np

def compute_ipw_weights(selected, covariates):
    """Compute inverse probability weights for selection correction."""
    model = LogisticRegression(max_iter=1000)
    model.fit(covariates, selected)
    prob_selected = model.predict_proba(covariates)[:, 1]
    weights = 1 / prob_selected
    return weights / np.mean(weights)  # Normalize
```

---

## Limitations

- Perfect selection correction requires knowing the true selection model
- Extreme weights can cause instability in estimates
- Unmeasured confounders of the selection process cannot be corrected
- Corrected estimates apply to the target population, not the sample

---

## Exercises

1. **DAG exercise**: Draw a DAG where conditioning on variable Z creates collider bias between X and Y. Under what conditions does the bias disappear?

2. **Simulation**: Simulate the Berkson bias scenario. Generate hospitalization as a function of smoking and a disease. Show that among hospitalized patients, smoking and disease appear negatively correlated.

3. **Critical appraisal**: Find a study that conditions on a potential collider. How does this affect the interpretation?

4. **Design**: You want to study the effect of a drug using electronic health records, but only patients who visited a doctor are in the database. How would you correct for this selection?

---

## Projects

### Project 1: Survivorship Bias Simulation
Create a simulation where products have random quality and marketing spend, only products above a sales threshold survive, and show that among survivors, marketing appears to cause success.

### Project 2: Selection Bias in Observational Studies
Find a published observational study that may have selection bias. Identify the selection mechanism, draw the DAG, assess the direction and magnitude of bias, and suggest corrections.
