---
title: "Lesson 4: Confounding"
subtitle: "From First Principles to Modern Causal Inference"
---

# Lesson 4: Confounding

## Opening Story

In 1986, Charles Freed and colleagues published a study in the New England Journal of Medicine that sent shockwaves through cardiology. They examined treatment data for kidney stones and found something seemingly impossible:

For **small stones**, Treatment A was better than Treatment B.
For **large stones**, Treatment A was better than Treatment B.
But **overall**, Treatment B was better than Treatment A.

This is Simpson's Paradox—the same data can tell opposite stories depending on how you look at it. The resolution lies in understanding confounding: the groups receiving different treatments were fundamentally different, and that difference distorted the overall comparison.

---

## Learning Objectives

By the end of this lesson, you should be able to:

1. Define confounding and identify when it occurs
2. Explain Simpson's Paradox and its resolution
3. Understand omitted variable bias
4. Distinguish between confounders, mediators, and colliders
5. Recognize post-treatment bias
6. Avoid over-adjustment

---

## 4.1 What is Confounding?

### The Definition

**Confounding** occurs when a variable (the confounder) is a common cause of both the treatment and the outcome, creating a non-causal association between them.

### The Confounding Triangle

```
Confounder (Z)
     ↓   ↘
Treatment (X) → Outcome (Y)
```

If we want to know whether $X$ causes $Y$, but $Z$ causes both $X$ and $Y$, then the association between $X$ and $Y$ is "confounded" by $Z$.

### Formal Definition

A variable $Z$ is a confounder if:
1. $Z$ is associated with $X$ (the treatment)
2. $Z$ is associated with $Y$ (the outcome)
3. $Z$ is not on the causal pathway from $X$ to $Y$

### Why Confounding Matters

If we ignore confounding, we get biased estimates of causal effects. The bias is the difference between what we estimate and the true causal effect.

**Mathematically**: If the true causal effect is $\tau$, but we estimate the association $\delta$, the bias is:
$$\text{Bias} = \delta - \tau$$

---

## 4.2 Simpson's Paradox

### The Kidney Stone Study

Let's examine the classic kidney stone data:

| | Treatment A | Treatment B |
|---|---|---|
| **Small Stones** | 93% success (81/87) | 87% success (234/270) |
| **Large Stones** | 73% success (192/263) | 69% success (55/80) |
| **Overall** | 78% success (273/350) | 83% success (289/350) |

Treatment A is better for small stones AND better for large stones, but worse overall. How is this possible?

### The Resolution

The key is understanding the treatment assignment:
- Treatment A (open surgery) was mostly given to severe cases (large stones)
- Treatment B (nephrolithotripsy) was mostly given to mild cases (small stones)

Stone size is a confounder: it affects both which treatment patients received AND their success rate.

### Visualizing Simpson's Paradox

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create the kidney stone data
data = {
    'stone_size': ['Small']*2 + ['Large']*2 + ['Small']*2 + ['Large']*2,
    'treatment': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
    'success': [93, 73, 87, 69, 234, 55, 192, 270],
    'total': [87, 263, 270, 80, 270, 80, 87, 263]
}

df = pd.DataFrame(data)
df['rate'] = df['success'] / df['total']

# Overall rates
overall = df.groupby('treatment')['success'].sum() / df.groupby('treatment')['total'].sum()
print("Overall success rates:")
print(overall)
print(f"\nTreatment B appears better overall: {overall['B']:.1%} vs {overall['A']:.1%}")

# But by stone size...
print("\nBy stone size:")
print(df.pivot_table(values='rate', index='stone_size', columns='treatment'))
print("\nTreatment A is better for both small AND large stones!")
```

### The Visual Explanation

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# By stone size (the truth)
sizes = ['Small', 'Large']
a_rates = [0.93, 0.73]
b_rates = [0.87, 0.69]

x = np.arange(len(sizes))
width = 0.35

axes[0].bar(x - width/2, a_rates, width, label='Treatment A', color='blue', alpha=0.7)
axes[0].bar(x + width/2, b_rates, width, label='Treatment B', color='red', alpha=0.7)
axes[0].set_xlabel('Stone Size')
axes[0].set_ylabel('Success Rate')
axes[0].set_title('By Stone Size: Treatment A is Better')
axes[0].set_xticks(x)
axes[0].set_xticklabels(sizes)
axes[0].legend()
axes[0].set_ylim(0, 1)

# Overall (confounded)
treatments = ['A', 'B']
overall_rates = [0.78, 0.83]

axes[1].bar(treatments, overall_rates, color=['blue', 'red'], alpha=0.7)
axes[1].set_xlabel('Treatment')
axes[1].set_ylabel('Success Rate')
axes[1].set_title('Overall: Treatment B Appears Better')
axes[1].set_ylim(0, 1)

# Add text
axes[1].text(0, 0.78 + 0.02, '78%', ha='center', fontweight='bold')
axes[1].text(1, 0.83 + 0.02, '83%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('../figures/04-simpsons-paradox.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Why This Happens

The key insight is that Treatment A was given disproportionately to severe cases:

```python
# Treatment assignment by stone size
print("Treatment assignment:")
print(df.pivot_table(values='total', index='stone_size', columns='treatment', aggfunc='sum'))

# Treatment A gets 75% of large stones (severe cases)
# Treatment B gets 77% of small stones (mild cases)
# This creates the paradox
```

---

## 4.3 Omitted Variable Bias

### The Bias Formula

When we omit a confounder $Z$ from our analysis, the bias in the estimated effect of $X$ on $Y$ is:

$$\text{Bias} = \beta_{XZ} \cdot \beta_{ZY}$$

where:
- $\beta_{XZ}$ is the effect of $Z$ on $X$ (how strongly $Z$ affects treatment)
- $\beta_{ZY}$ is the effect of $Z$ on $Y$ (how strongly $Z$ affects outcome)

### The Direction of Bias

The bias can be:
- **Positive** if both effects have the same sign
- **Negative** if the effects have opposite signs
- **Zero** if either effect is zero

### Example: Education and Earnings

```python
# Simulate omitted variable bias
np.random.seed(42)
n = 1000

# Ability (omitted confounder)
ability = np.random.normal(0, 1, n)

# Education depends on ability
education = 12 + 2 * ability + np.random.normal(0, 1, n)

# Earnings depend on education and ability
earnings = 20000 + 5000 * education + 10000 * ability + np.random.normal(0, 5000, n)

# True effect of education on earnings: 5000

# Omitted variable regression (ignoring ability)
from sklearn.linear_model import LinearRegression

reg_omitted = LinearRegression().fit(education.reshape(-1, 1), earnings)
effect_omitted = reg_omitted.coef_[0]

# Complete regression (including ability)
reg_complete = LinearRegression().fit(
    np.column_stack([education, ability]), earnings
)
effect_complete = reg_complete.coef_[0]

print(f"True effect of education: 5000")
print(f"Effect ignoring ability: {effect_omitted:.2f}")
print(f"Effect controlling for ability: {effect_complete:.2f}")
print(f"\nOmitted variable bias: {effect_omitted - 5000:.2f}")
```

### The Bias in the Education Example

- Ability positively affects education ($\beta_{XZ} > 0$)
- Ability positively affects earnings ($\beta_{ZY} > 0$)
- Therefore, omitted variable bias is positive
- We overestimate the effect of education on earnings

---

## 4.4 Confounders vs. Mediators vs. Colliders

### The Key Distinction

| Variable Type | Definition | Should We Control? |
|---------------|------------|-------------------|
| **Confounder** | Common cause of X and Y | Yes |
| **Mediator** | On the causal pathway from X to Y | Depends on estimand |
| **Collider** | Common effect of X and Y | No |

### Confounders

```
Z
↓   ↘
X → Y
```

- $Z$ causes both $X$ and $Y$
- The association between $X$ and $Y$ is confounded by $Z$
- We should control for $Z$

### Mediators

```
X → Z → Y
```

- $Z$ is on the causal pathway from $X$ to $Y$
- If we want the **total effect**, don't control for $Z$
- If we want the **direct effect**, control for $Z$

### Colliders

```
X → Z ← Y
```

- Both $X$ and $Y$ cause $Z$
- Controlling for $Z$ creates spurious association
- We should NOT control for $Z$

### The Danger of Controlling for Everything

Many practitioners think "the more variables, the better." This is wrong. Controlling for mediators and colliders can introduce bias where none existed.

---

## 4.5 Post-Treatment Bias

### The Problem

Post-treatment bias occurs when you control for a variable that is affected by the treatment. This can create the appearance of no effect even when there is one.

### Example: The Clinical Trial

```python
# Simulate post-treatment bias
np.random.seed(42)
n = 1000

# Treatment
T = np.random.binomial(1, 0.5, n)

# Potential outcomes
Y0 = np.random.normal(100, 20, n)
Y1 = Y0 + 10  # True effect: 10

# Observed outcome
Y = Y1 * T + Y0 * (1 - T)

# Post-treatment variable (affected by treatment)
Z = 50 + 3 * T + np.random.normal(0, 10, n)

# Naive estimate (ignoring post-treatment variable)
effect_naive = Y[T==1].mean() - Y[T==0].mean()

# Biased estimate (controlling for post-treatment variable)
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(np.column_stack([T, Z]), Y)
effect_biased = reg.coef_[0]

print(f"True effect: 10")
print(f"Naive estimate: {effect_naive:.2f}")
print(f"Controlling for post-treatment variable: {effect_biased:.2f}")
print(f"\nControlling for a post-treatment variable introduces bias!")
```

### Why This Happens

When we control for $Z$ (which is affected by treatment), we're comparing:
- Treated units with a certain level of $Z$
- Control units with the same level of $Z$

But since treatment affects $Z$, this comparison is no longer apples-to-apples. We're essentially holding part of the treatment effect constant.

---

## 4.6 Over-Adjustment

### The Problem

Over-adjustment occurs when we control for variables that shouldn't be controlled for, typically mediators or colliders.

### Example: The Job Training Program

A job training program might work through:
1. Training → Skills → Earnings
2. Training → Confidence → Earnings

If we want the total effect of training on earnings, we should NOT control for skills or confidence, because they are mediators.

### When to Control for What

| Variable Type | Total Effect | Direct Effect |
|---------------|--------------|---------------|
| Confounder | Control | Control |
| Mediator | Don't control | Control |
| Collider | Don't control | Don't control |

---

## 4.7 Python Workshop: Diagnosing Confounding

### Using DAGs to Identify Confounders

```python
import networkx as nx
import matplotlib.pyplot as plt

# Education and earnings DAG
G = nx.DiGraph()
G.add_edges_from([
    ('Ability', 'Education'),
    ('Ability', 'Earnings'),
    ('Education', 'Earnings'),
    ('Family', 'Education'),
    ('Family', 'Earnings')
])

plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, seed=42)
node_colors = ['#e76f51' if node in {'Ability', 'Family'} else '#9dd9d2' for node in G.nodes]
nx.draw(G, pos, ax=plt.gca(), with_labels=True, node_size=3000,
        node_color=node_colors, font_size=12, font_weight='bold',
        arrowsize=20)

plt.title('DAG: Education and Earnings')
plt.show()

print("Confounders to control for: Ability, Family")
print("These are common causes of Education and Earnings")
```

### Sensitivity Analysis for Unmeasured Confounding

```python
# Rosenbaum bounds for sensitivity analysis
def rosenbaum_bound(gamma, n_treated, n_control):
    """
    Calculate the sensitivity parameter gamma.
    Gamma = 1 means no unmeasured confounding.
    """
    # Simplified calculation
    # In practice, use more sophisticated methods
    return gamma

# Example: How strong would unmeasured confounding need to be?
print("Sensitivity Analysis:")
print("If our estimated effect is 5.0, how strong would")
print("unmeasured confounding need to be to explain it away?")

# Calculate E-value (simplified)
estimated_effect = 5.0
null_effect = 0.0

# E-value: minimum strength of association that an unmeasured
# confounder would need to have with both treatment and outcome
# to explain away the observed effect
e_value = estimated_effect + np.sqrt(estimated_effect * (estimated_effect + 4))
print(f"\nE-value: {e_value:.2f}")
print("An unmeasured confounder would need to be associated")
print(f"with both treatment and outcome by a factor of {e_value:.2f}")
print("to explain away the observed effect.")
```

---

## 4.8 Case Study: Hormone Replacement Therapy

### The Observational Finding

In the 1990s, observational studies consistently found that women taking hormone replacement therapy (HRT) had lower rates of heart disease. This led many doctors to recommend HRT.

### The Problem

The Women's Health Initiative (WHI) conducted a randomized trial and found the opposite: HRT actually *increased* heart disease risk.

### What Went Wrong?

The observational studies were confounded by **socioeconomic status**:
- Wealthier women were more likely to take HRT
- Wealthier women had better health outcomes overall
- This created a spurious negative association between HRT and heart disease

### The Lesson

Even well-designed observational studies can be confounded by unmeasured or poorly measured variables. This is why randomized experiments are the gold standard.

---

## 4.9 Common Mistakes

### Mistake 1: Controlling for Everything

**The error**: Including every available variable in a regression model.

**Why it's wrong**: Controlling for mediators and colliders introduces bias.

**How to avoid it**: Draw a DAG and only control for confounders.

### Mistake 2: Ignoring Unmeasured Confounders

**The error**: Assuming that controlling for observed variables is sufficient.

**Why it's wrong**: Unmeasured confounders can still bias results.

**How to address it**: Conduct sensitivity analysis.

### Mistake 3: Confusing Statistical with Practical Significance

**The error**: Assuming that a statistically significant result is practically important.

**Why it matters**: With large samples, tiny effects can be statistically significant.

**How to avoid it**: Report effect sizes and confidence intervals.

---

## 4.10 Discussion Questions

1. **Education and Income**: A study finds that college graduates earn more than non-graduates. What confounders might explain this association? How could you test whether education causally increases income?

2. **Medical Decision**: A doctor observes that patients who take a certain supplement have better health outcomes. Should the doctor recommend this supplement? What else would you want to know?

3. **Policy Evaluation**: A city implements a new traffic safety program and accidents decrease. The mayor claims the program "caused" the reduction. What confounding might exist?

4. **The Collider Problem**: In hiring, we observe only the interview scores of people who were hired. What happens to the correlation between ability and charisma when we condition on being hired?

5. **Sensitivity Analysis**: How would you assess whether unmeasured confounding could explain your results?

---

## 4.11 Interview Questions

### Question 1
**"What is Simpson's Paradox and how do you resolve it?"**

**Model Answer**: Simpson's Paradox occurs when a trend appears in several groups of data but disappears or reverses when the groups are combined. The resolution is to identify the confounding variable that explains the reversal. In the kidney stone example, treatment assignment was confounded by stone size: severe cases got one treatment, mild cases got another. When we stratify by stone size, Treatment A is better in both groups. The lesson is that combining data without accounting for confounders can lead to misleading conclusions.

### Question 2
**"What is the difference between a confounder and a mediator?"**

**Model Answer**: A confounder is a common cause of treatment and outcome (Z causes both X and Y). A mediator is on the causal pathway from treatment to outcome (X causes Z, which causes Y). We should control for confounders to estimate causal effects, but we should not control for mediators if we want the total effect. For example, in studying whether education increases earnings, "ability" is a confounder (affects both education and earnings), while "skills" is a mediator (education increases skills, which increase earnings).

### Question 3
**"Why is controlling for more variables not always better?"**

**Model Answer**: Controlling for variables that shouldn't be controlled for can introduce bias. Specifically: (1) conditioning on colliders creates spurious associations, (2) conditioning on mediators removes part of the causal effect, and (3) conditioning on post-treatment variables can create the appearance of no effect. The correct approach is to draw a DAG and only control for confounders—variables that are common causes of treatment and outcome.

### Question 4
**"What is omitted variable bias and how can you detect it?"**

**Model Answer**: Omitted variable bias occurs when a confounder is left out of the analysis. The bias equals the effect of the omitted variable on treatment times its effect on outcome. You can detect potential omitted variable bias by: (1) drawing a DAG to identify potential confounders, (2) comparing estimates with and without the variable, (3) conducting sensitivity analysis to see how strong unmeasured confounding would need to be to explain away results.

### Question 5
**"Design a study to evaluate whether smoking causes lung cancer."**

**Model Answer**: Since we can't randomize smoking, we'd use observational methods with careful confounding control. Key steps: (1) Draw a DAG identifying confounders (genetics, stress, diet), (2) Use large cohort studies with detailed covariate data, (3) Apply propensity score matching or regression adjustment, (4) Use instrumental variables if possible (e.g., cigarette taxes), (5) Conduct sensitivity analysis for unmeasured confounding, (6) Check consistency with biological mechanisms and animal studies.

---

## 4.12 Knowledge Check

### Multiple Choice

1. **A confounder is:**
   - A) A variable affected by treatment
   - B) A variable on the causal pathway
   - C) A common cause of treatment and outcome
   - D) A common effect of treatment and outcome

2. **Simpson's Paradox is resolved by:**
   - A) Combining all data
   - B) Identifying and stratifying by the confounder
   - C) Ignoring the paradox
   - D) Using larger samples

3. **We should NOT control for:**
   - A) Confounders
   - B) Mediators (for total effect)
   - C) Colliders
   - D) Both B and C

4. **Omitted variable bias equals:**
   - A) Effect of omitted on treatment × Effect of omitted on outcome
   - B) Effect of treatment on outcome
   - C) Effect of outcome on treatment
   - D) None of the above

5. **Post-treatment bias occurs when:**
   - A) We control for a baseline variable
   - B) We control for a variable affected by treatment
   - C) We don't control for any variables
   - D) We use a randomized experiment

### Short Answer

6. **Explain why controlling for a collider introduces bias.**

7. **Give an example of omitted variable bias in a real-world study.**

8. **What is the difference between the total effect and the direct effect?**

9. **How can sensitivity analysis help with unmeasured confounding?**

10. **Draw a DAG showing confounding in the relationship between exercise and heart disease.**

---

## Watch and Connect

```{admonition} Recommended lecture
:class: tip

[**Backdoor adjustment and structural causal models**](https://www.youtube.com/watch?v=dB8r4Afmobo&list=PLoazKTcS0Rzb6bb9L508cyJ1z-U9iWkA0&index=28)

Connects confounding control to a precise graphical adjustment rule.

**Active-viewing prompt:** Identify one variable that should be adjusted for and one that must not be adjusted for.
```

---

## 4.13 Summary

In this lesson, we learned:

1. **Confounding** occurs when a common cause creates spurious associations
2. **Simpson's Paradox** shows how confounding can reverse apparent trends
3. **Omitted variable bias** quantifies the effect of missing confounders
4. **Confounders, mediators, and colliders** require different handling
5. **Post-treatment bias** occurs when controlling for treatment effects
6. **Over-adjustment** can introduce bias where none existed
7. **DAGs** help identify what to control for (and what not to)

The key lesson: **draw a DAG, identify confounders, and only control for those.**

---

## 4.14 Further Reading

### Classic Works
- Simpson, E.H. (1951). "The Interaction between Individuals in a Two-Way Table." *Journal of the Royal Statistical Society*.
- Berkson, J. (1946). "Limitations of the Application of Fourfold Table Analysis to Hospital Data." *Biometrics Bulletin*.
- Greenland, S. & Robins, J.M. (1986). "Identifiability, Exchangeability and Epidemiological Confounding." *International Journal of Epidemiology*.

### Modern Textbooks
- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*. Cambridge University Press.
- Hernán, M.A. & Robins, J.M. (2020). *Causal Inference: What If*. Chapman and Hall/CRC.
- Cunningham, S. (2021). *Causal Inference: The Mixtape*. MIT Press.


---

## Worked Examples

### Example 1: Identifying Confounders

A confounder is a variable that causes both the treatment and the outcome, creating a spurious association.

```python
import numpy as np
import pandas as pd
from scipy import stats

np.random.seed(42)
n = 1000

# Confounder affects both treatment and outcome
socioeconomic = np.random.normal(0, 1, n)
treatment = (0.5 * socioeconomic + np.random.normal(0, 0.5, n)) > 0
outcome = 2.0 * socioeconomic + np.random.normal(0, 1, n)

# Naive estimate (confounded)
naive_effect = np.mean(outcome[treatment]) - np.mean(outcome[~treatment])
print(f"Naive estimate: {naive_effect:.3f}")

# Adjusted estimate (controlling for confounder)
import statsmodels.api as sm
X = sm.add_constant(np.column_stack([treatment.astype(int), socioeconomic]))
model = sm.OLS(outcome, X).fit()
adjusted_effect = model.params[1]
print(f"Adjusted estimate: {adjusted_effect:.3f}")
print(f"True effect: 0.000 (no treatment effect)")
```

### Example 2: Backdoor Criterion

The backdoor criterion says: block all backdoor paths (paths with an arrow into the treatment) by conditioning on an appropriate set of variables.

```python
import networkx as nx

# DAG: Z → T → Y, Z → Y (Z is a confounder)
G = nx.DiGraph()
G.add_edges_from([('Z', 'T'), ('T', 'Y'), ('Z', 'Y')])

# Backdoor paths from T to Y
# T ← Z → Y is a backdoor path
# Conditioning on Z blocks this path

print("Backdoor paths from T to Y:")
for path in nx.all_simple_paths(G.to_undirected(), 'T', 'Y'):
    if 'T' in path and 'Y' in path:
        print(f"  {' → '.join(path)}")

print("\nConditioning on Z blocks the backdoor path")
```


---

## Exercises

### Exercise 1

For a study of exercise (T) on heart disease (Y), identify at least 3 potential confounders and explain why they satisfy the definition.

### Exercise 2

Draw a DAG where the backdoor criterion requires conditioning on a mediator. Explain why this is problematic.

### Exercise 3

Write Python code to demonstrate how omitted variable bias changes when you fail to control for a confounder.
