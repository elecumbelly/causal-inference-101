---
title: "Lesson 6: Selection Bias & Collider Bias"
subtitle: "From First Principles to Modern Causal Inference"
---

# Lesson 6: Selection Bias & Collider Bias

## Opening Story: Berkson's Paradox

In 1946, Joseph Berkson noticed something strange in hospital data. Among patients admitted to the Mayo Clinic, those with gallstones appeared less likely to have diabetes than expected. This was puzzling because gallstones and diabetes are both common conditions.

Berkson realized the problem: he was conditioning on being in the hospital. Both gallstones and diabetes increase the probability of hospitalization. By looking only at hospitalized patients, he was conditioning on a common effect of two independent causes—which made them appear negatively correlated.

This is Berkson's paradox, and it illustrates the fundamental problem with collider bias: conditioning on a common effect of two variables creates a spurious association between them, even if they are completely independent.

---

## Learning Objectives

By the end of this lesson, you should be able to:

1. Define collider bias and explain when it occurs
2. Distinguish between different types of selection bias
3. Identify M-bias structures
4. Recognize sampling and survivor bias
5. Design studies to avoid selection bias

---

## 6.1 Collider Bias

### The Definition

A **collider** is a variable that is causally affected by two other variables. In DAG notation, a collider on the path between $X$ and $Y$ is a variable $Z$ such that $X \rightarrow Z \leftarrow Y$.

### The Key Insight

When we condition on (control for, stratify by, or select based on) a collider, we create a spurious association between its causes.

### Why This Happens

Consider two independent diseases, A and B. If we condition on being in the hospital (which is caused by having either disease), we create a negative association between A and B:
- Among hospitalized patients, those with A are less likely to also have B (because they were hospitalized for A, not B)
- This creates the illusion that A and B are negatively correlated

### Mathematical Demonstration

```python
import numpy as np
import pandas as pd

np.random.seed(42)
n = 10000

# Two independent diseases
disease_a = np.random.binomial(1, 0.1, n)
disease_b = np.random.binomial(1, 0.1, n)

# Hospital admission (caused by either disease)
prob_hospital = 0.05 + 0.3 * disease_a + 0.3 * disease_b
in_hospital = np.random.binomial(1, prob_hospital)

# In general population: independent
corr_population = np.corrcoef(disease_a, disease_b)[0, 1]

# In hospital: negatively correlated (collider bias)
hospital_mask = in_hospital == 1
corr_hospital = np.corrcoef(disease_a[hospital_mask], disease_b[hospital_mask])[0, 1]

print(f"Correlation in general population: {corr_population:.3f}")
print(f"Correlation in hospital: {corr_hospital:.3f}")
print(f"\nConditioning on hospital admission creates spurious negative correlation!")
```

---

## 6.2 Types of Selection Bias

### 1. Selection on the Dependent Variable

When we condition on the outcome variable, we create collider bias.

**Example**: Studying the causes of success among successful people. By conditioning on success, we may create spurious associations between its causes.

### 2. Berkson's Bias

When we condition on a common effect of two independent variables, they appear correlated.

**Example**: In a case-control study, if we select cases (people with the disease) and controls (people without the disease), we are conditioning on disease status, which may be a collider.

### 3. Healthy Worker Effect

Workers are healthier than the general population because unhealthy people can't work. If we study only workers, we underestimate disease rates.

### 4. Attrition Bias

When participants drop out of a study, the remaining sample may not be representative. If dropout is related to treatment or outcomes, results are biased.

---

## 6.3 M-Bias

### The Structure

M-bias occurs when conditioning on a variable that is on the causal pathway between two confounders, creating bias where none existed.

```
U1       U2
 ↓   M   ↓
 X → Z → Y
```

In this DAG:
- $U1$ confounds the $X$-$Z$ relationship
- $U2$ confounds the $Z$-$Y$ relationship
- Conditioning on $M$ (which is between $U1$ and $U2$) can introduce bias

### The Lesson

Not all variables that look like confounders should be controlled for. The DAG determines what to condition on.

---

## 6.4 Sampling Bias

### Definition

Sampling bias occurs when the way we select our sample creates a non-representative sample.

### Types

1. **Self-selection bias**: Participants choose whether to be in the study
2. **Volunteer bias**: Volunteers differ from non-volunteers
3. **Convenience bias**: Easy-to-reach participants differ from the population
4. **Survival bias**: Only survivors are observed (e.g., WWII bomber armor example)

### The WWII Bomber Example

During WWII, the military examined returning bombers and found more bullet holes in the wings and fuselage than in the engines. They wanted to add armor to the wings and fuselage.

Abraham Wald pointed out the flaw: they were only looking at planes that survived. The planes hit in the engines didn't return. The absence of bullet holes in the engines among survivors was evidence that engine hits were the most dangerous.

---

## 6.5 Python Workshop: Simulating Selection Bias

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
n = 5000

# Simulate a study of the effect of a drug on recovery
# Drug assignment is random, but recovery depends on severity
severity = np.random.normal(0, 1, n)
drug = np.random.binomial(1, 0.5, n)
recovery_prob = 0.5 + 0.2 * drug - 0.3 * severity
recovered = np.random.binomial(1, recovery_prob)

# Create dataframe
df = pd.DataFrame({
    'severity': severity,
    'drug': drug,
    'recovered': recovered
})

# True effect: drug increases recovery probability by 0.2
print("True drug effect on recovery probability: 0.20")

# Overall estimate (no selection)
effect_overall = df[df['drug']==1]['recovered'].mean() - df[df['drug']==0]['recovered'].mean()
print(f"Overall estimate: {effect_overall:.3f}")

# Estimate among recovered patients only (selection on outcome!)
recovered_df = df[df['recovered'] == 1]
effect_recovered = recovered_df[recovered_df['drug']==1]['severity'].mean() - \
                   recovered_df[recovered_df['drug']==0]['severity'].mean()
print(f"\nEstimate among recovered patients: {effect_recovered:.3f}")
print("(This is biased because we're conditioning on the outcome)")

# Estimate among severe cases (potential selection bias)
severe_df = df[df['severity'] > 0]
effect_severe = severe_df[severe_df['drug']==1]['recovered'].mean() - \
                severe_df[severe_df['drug']==0]['recovered'].mean()
print(f"\nEstimate among severe cases: {effect_severe:.3f}")
```

---

## 6.6 Case Study: Berkson's Paradox in Hiring

### The Scenario

A tech company finds that among their employees, coding ability and communication skills appear negatively correlated. They conclude that "great coders are bad communicators."

### The Collider Bias Explanation

The company only hires people who are good at coding OR good at communication (or both). By conditioning on being hired, they create a collider bias:
- People hired primarily for coding are less likely to be great communicators
- People hired primarily for communication are less likely to be great coders

### The Lesson

Among the general population, coding ability and communication skills might be positively correlated (smart people tend to be good at both). But among employees, the correlation appears negative because of selection.

---

## 6.7 Common Mistakes

1. **Conditioning on colliders**: Always draw the DAG before deciding what to control for
2. **Ignoring selection bias**: Consider who is in your sample and why
3. **Survivor bias**: Remember that non-survivors are missing from the data
4. **Attrition bias**: Track all participants, not just completers

---

## 6.8 Discussion Questions

1. **Berkson's Paradox**: A study finds that among successful people, health and wealth are negatively correlated. Is this causal? What selection might be at play?

2. **Survivor Bias**: A company studies only long-term employees and finds that job satisfaction predicts longevity. What might they be missing?

3. **Collider in Medicine**: A researcher studies only patients who visited a doctor and finds that smoking and exercise are negatively correlated. Is this real?

4. **Hiring Practices**: If a company hires based on a test that combines two skills, what will they observe about the correlation between those skills among employees?

5. **Study Design**: How can you design a study to minimize selection bias?

---

## 6.9 Knowledge Check

### Multiple Choice

1. **A collider is:**
   - A) A common cause of X and Y
   - B) A common effect of X and Y
   - C) A mediator between X and Y
   - D) A confounder

2. **Conditioning on a collider:**
   - A) Blocks spurious associations
   - B) Creates spurious associations
   - C) Has no effect
   - D) Always reduces bias

3. **Berkson's paradox occurs when:**
   - A) We condition on a common cause
   - B) We condition on a common effect
   - C) We don't condition on anything
   - D) We have a large sample

4. **Survivor bias occurs when:**
   - A) We study everyone
   - B) We only study those who "survived" a selection process
   - C) We randomize treatment
   - D) We control for confounders

5. **Selection bias is most problematic when:**
   - A) It's random
   - B) It's related to both treatment and outcome
   - C) It's related to neither
   - D) The sample is large

### Short Answer

6. **Explain why conditioning on hospital admission creates a spurious correlation between independent diseases.**

7. **Give an example of survivor bias in business research.**

8. **How can you detect potential collider bias in an observational study?**

9. **What is the healthy worker effect and why does it matter for occupational health studies?**

10. **Design a study to avoid selection bias when studying the effect of a new educational program.**

---

## Watch and Connect

```{admonition} Recommended lecture
:class: tip

[**Graphical models: colliders and selection**](https://www.youtube.com/watch?v=Go4EkHN_PcA&list=PLoazKTcS0Rzb6bb9L508cyJ1z-U9iWkA0&index=19)

The graphical-model lecture makes collider bias visible as a path-opening problem.

**Active-viewing prompt:** Draw the selection variable explicitly before deciding what to condition on.
```

---

## 6.10 Summary

In this lesson, we learned:

1. **Collider bias** occurs when conditioning on common effects of independent variables
2. **Berkson's paradox** illustrates how hospital data can create spurious associations
3. **Selection bias** arises from non-random inclusion in the study sample
4. **Survivor bias** occurs when we only observe those who "survived" a selection process
5. **M-bias** shows that controlling for the wrong variables can introduce bias

---

## 6.11 Further Reading

- Berkson, J. (1946). "Limitations of the Application of Fourfold Table Analysis to Hospital Data." *Biometrics Bulletin*.
- Hernán, M.A., Hernández-Díaz, S., & Robins, J.M. (2004). "A Structural Approach to Selection Bias." *Epidemiology*.
- Elwert, F. & Winship, C. (2014). "Endogenous Selection Bias." *Annual Review of Sociology*.


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
