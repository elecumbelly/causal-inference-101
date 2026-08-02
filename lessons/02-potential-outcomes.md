---
title: "Lesson 2: Potential Outcomes"
subtitle: "From First Principles to Modern Causal Inference"
---

# Lesson 2: Potential Outcomes

## Opening Story

In the mid-1970s, the National Supported Work Demonstration randomly assigned eligible workers to a subsidized employment programme or a control group. Robert LaLonde’s influential 1986 analysis later used those experimental results as a benchmark for evaluating observational estimators.

Some people assigned to training didn't show up. Some people in the control group found training elsewhere. And when researchers tried to evaluate the program using standard observational methods, they got wildly different answers depending on which data they used.

LaLonde compared the randomized benchmark with estimates formed by replacing the experimental controls with non-experimental comparison groups. Many conventional observational specifications failed to recover the benchmark—an enduring lesson that adjustment cannot compensate for poor design or inadequate overlap.

This lesson introduces the framework that makes this kind of evaluation possible: the **Potential Outcomes Framework**, also known as the **Rubin Causal Model**.

---

## Learning Objectives

By the end of this lesson, you should be able to:

1. Define potential outcomes and the Rubin Causal Model
2. Explain the Fundamental Problem of Causal Inference
3. Calculate and interpret ATE, ATT, and ATC
4. Describe SUTVA and its importance
5. Understand exchangeability and when it holds
6. Explain consistency and positivity
7. Implement potential outcomes reasoning in Python

---

```{figure} ../figures/instructional/potential-outcomes.svg
:name: lesson-02-potential-outcomes
:alt: One potential outcome is observed; the other is the missing counterfactual.
:width: 100%

One potential outcome is observed; the other is the missing counterfactual.
```

---

## 2.1 The Rubin Causal Model

### The Framework

The Potential Outcomes Framework, developed by Donald Rubin starting in the 1970s, provides a rigorous mathematical foundation for causal inference.

**Core Idea**: For each unit $i$, define two potential outcomes:
- $Y_i(1)$: The outcome if unit $i$ receives treatment
- $Y_i(0)$: The outcome if unit $i$ does not receive treatment

The **individual causal effect** for unit $i$ is:
$$\tau_i = Y_i(1) - Y_i(0)$$

This is the difference between what happened under treatment and what would have happened under control.

### The Notation

Let's establish consistent notation throughout this book:

| Symbol | Meaning |
|--------|---------|
| $i$ | Unit index (person, firm, country, etc.) |
| $T_i$ | Treatment assignment for unit $i$ (1 = treated, 0 = control) |
| $Y_i$ | Observed outcome for unit $i$ |
| $Y_i(1)$ | Potential outcome under treatment |
| $Y_i(0)$ | Potential outcome under control |
| $\tau_i$ | Individual treatment effect: $Y_i(1) - Y_i(0)$ |
| $N$ | Sample size |

### The Fundamental Problem

We can only observe one potential outcome for each unit. If $T_i = 1$, we observe $Y_i(1)$ but not $Y_i(0)$. If $T_i = 0$, we observe $Y_i(0)$ but not $Y_i(1)$.

The unobserved outcome is the **counterfactual**—what would have happened under the alternative treatment.

This is why causal inference is fundamentally about **missing data**.

---

## 2.2 Average Treatment Effects

Since we can't estimate individual causal effects (we only see one outcome per unit), we work with **population-level quantities**.

### Average Treatment Effect (ATE)

The ATE is the average effect across the entire population:

$$ATE = E[Y(1) - Y(0)] = E[Y(1)] - E[Y(0)]$$

This tells us: "On average, what is the effect of treatment across everyone?"

### Average Treatment Effect on the Treated (ATT)

The ATT is the average effect among those who actually received treatment:

$$ATT = E[Y(1) - Y(0) | T = 1]$$

This tells us: "For people who actually got treated, what was the average effect?"

### Average Treatment Effect on the Untreated (ATC)

The ATC is the average effect among those who did not receive treatment:

$$ATC = E[Y(1) - Y(0) | T = 0]$$

This tells us: "For people who didn't get treated, what would the average effect have been?"

### Relationship Between ATE, ATT, and ATC

The ATE is a weighted average of ATT and ATC:

$$ATE = P(T=1) \cdot ATT + P(T=0) \cdot ATC$$

---

## 2.3 Key Assumptions

### SUTVA (Stable Unit Treatment Value Assumption)

SUTVA has two components:

1. **No interference**: One unit's treatment doesn't affect another unit's outcome
   - If I take a drug, it doesn't affect your health outcome
   - If I get a vaccine, it doesn't change your infection risk (ignoring herd immunity)

2. **Treatment variation irrelevance**: There's only one version of each treatment level
   - "Treatment" is precisely defined
   - All treated units receive the same treatment

**Why SUTVA matters**: Without it, the potential outcomes framework breaks down. If my treatment affects your outcome, then your potential outcomes depend on what treatment I receive, and the whole framework becomes much more complex.

**When SUTVA fails**:
- Contagious diseases (treatment affects others' outcomes)
- Spillovers in economic programs (my benefit affects yours)
- Network effects (my behavior influences yours)

### Exchangeability

Exchangeability (also called unconfoundedness) means that treatment assignment is independent of potential outcomes, conditional on observed covariates:

$$(Y(0), Y(1)) \perp T | X$$

**What this means**: If we compare two people with the same values of $X$, the one who got treated is comparable to the one who didn't. The treatment assignment is as good as random, conditional on $X$.

**When exchangeability holds**:
- In a properly randomized experiment (unconditionally)
- In an observational study where all confounders are measured and controlled

**When exchangeability fails**:
- When there are unmeasured confounders
- When treatment assignment depends on unobserved factors

**Why exchangeability is crucial**: It allows us to estimate causal effects by comparing treated and control groups. Without exchangeability, any differences in outcomes could be due to confounders rather than treatment.

### Consistency

Consistency means that the observed outcome equals the potential outcome corresponding to the observed treatment:

$$\text{If } T_i = t, \text{ then } Y_i = Y_i(t)$$

**What this means**: If someone received treatment, their observed outcome is their treatment potential outcome. If someone didn't receive treatment, their observed outcome is their control potential outcome.

**When consistency fails**:
- Multiple versions of treatment exist (different doses, delivery methods)
- Treatment is not well-defined
- There are multiple treatment levels and the mapping is ambiguous

### Positivity

Positivity (also called overlap) means that every unit has a positive probability of receiving either treatment level:

$$0 < P(T = 1 | X) < 1 \quad \text{for all } X$$

**What this means**: For every combination of covariate values, there's a chance of being treated and a chance of being in control. No one is deterministically assigned to treatment based on their covariates.

**When positivity fails**:
- Everyone with certain characteristics gets treated (e.g., only severe cases get a drug)
- Some subgroups have no one in treatment or control
- Perfect prediction of treatment from covariates

**Why positivity matters**: Without it, we can't estimate causal effects for certain subgroups because we have no comparison group.

---

## 2.4 The Connection to Randomized Experiments

### Why Randomization Works

In a randomized experiment:
- Treatment assignment $T$ is independent of potential outcomes $(Y(0), Y(1))$
- This is exchangeability **by design** (no conditioning needed)
- Therefore: $E[Y(0) | T = 1] = E[Y(0) | T = 0]$

This means we can estimate the ATE as:

$$ATE = E[Y | T = 1] - E[Y | T = 0]$$

The difference in observed outcomes equals the causal effect because randomization ensures the groups are comparable.

### The Lalonde Experiment

The National Supported Work (NSW) demonstration randomly assigned participants to job training or control. This gives us a benchmark: the experimental estimate of the training effect.

Lalonde then compared this experimental estimate to what various observational methods would have given using comparison groups from surveys (CPS, PSID) that weren't randomized.

Key finding: Many observational methods failed to recover the experimental effect, highlighting the importance of proper causal inference methods.

---

## 2.5 Mathematical Framework

### Deriving the ATE from Observational Data

Under exchangeability, we can derive:

$$E[Y | T = 1] = E[Y(1) | T = 1] = E[Y(1)]$$

$$E[Y | T = 0] = E[Y(0) | T = 0] = E[Y(0)]$$

Therefore:

$$ATE = E[Y(1)] - E[Y(0)] = E[Y | T = 1] - E[Y | T = 0]$$

This is why the simple difference in means works in a randomized experiment.

### When Exchangeability Doesn't Hold

If exchangeability doesn't hold (unmeasured confounding), then:

$$E[Y | T = 1] \neq E[Y(1)]$$
$$E[Y | T = 0] \neq E[Y(0)]$$

And the difference in means is biased:

$$E[Y | T = 1] - E[Y | T = 0] \neq ATE$$

The bias is:

$$\text{Bias} = \{E[Y(0) | T = 1] - E[Y(0) | T = 0]\}$$

This is the difference in baseline outcomes between treated and control groups—the **selection bias**.

---

## 2.6 Case Study: The Lalonde Job Training Experiment

### Background

The National Supported Work (NSW) Demonstration was a randomized job training program conducted in the mid-1970s. It targeted disadvantaged workers, including former substance abusers, welfare recipients, and ex-convicts.

### The Experiment

- 185 participants were randomly assigned to treatment (job training)
- 260 participants were randomly assigned to control
- Outcome: Real earnings in 1978

### The Key Finding

The experimental estimate showed that job training increased earnings by about $1,794 (in 1978 dollars).

But when researchers tried to replicate this using observational comparison groups (from CPS or PSID surveys), they got wildly different results—some methods suggested the program was harmful.

### Why This Matters

The Lalonde study demonstrated that:
1. Randomization provides a benchmark for causal inference
2. Observational methods can fail dramatically
3. The choice of comparison group matters enormously
4. Causal inference requires careful attention to assumptions

---

## 2.7 Python Workshop: Simulating Potential Outcomes

Let's implement the potential outcomes framework in Python.

### Setup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

np.random.seed(42)
```

### Simulating Potential Outcomes

```python
# Simulate a job training experiment
n_treated = 185
n_control = 260
n_total = n_treated + n_control

# Generate baseline characteristics (confounders)
# Lower income → more likely to be in training program
baseline_income = np.random.normal(10000, 5000, n_total)
baseline_income = np.maximum(baseline_income, 0)

# Education (years)
education = np.random.normal(10, 2, n_total)
education = np.clip(education, 6, 18)

# Age
age = np.random.normal(35, 10, n_total)
age = np.clip(age, 18, 65)

# Treatment assignment (random in experiment)
T = np.concatenate([np.ones(n_treated), np.zeros(n_control)])

# Potential outcomes
# Training effect depends on baseline characteristics
training_effect = 500 + 0.1 * baseline_income - 50 * education + 20 * age

Y0 = 1000 + 0.3 * baseline_income + 100 * education + 50 * age + np.random.normal(0, 1000, n_total)
Y1 = Y0 + training_effect

# Observed outcome
Y = Y1 * T + Y0 * (1 - T)

# Create dataframe
df = pd.DataFrame({
    'T': T,
    'Y': Y,
    'Y0': Y0,
    'Y1': Y1,
    'baseline_income': baseline_income,
    'education': education,
    'age': age,
    'training_effect': training_effect
})

print("Sample size:", len(df))
print("Treated:", int(T.sum()))
print("Control:", int((1-T).sum()))
```

### Estimating the ATE

```python
# True ATE (using potential outcomes)
true_ate = (Y1 - Y0).mean()
print(f"True ATE: ${true_ate:.2f}")

# Experimental estimate (difference in means)
ate_experimental = Y[T==1].mean() - Y[T==0].mean()
print(f"Experimental estimate: ${ate_experimental:.2f}")

# The experimental estimate should be close to the true ATE
print(f"Estimation error: ${abs(ate_experimental - true_ate):.2f}")
```

### Estimating ATT and ATC

```python
# ATT: Average effect among treated
att_true = (Y1[T==1] - Y0[T==1]).mean()
att_experimental = Y[T==1].mean() - Y0[T==1].mean()  # Using true Y0

print(f"True ATT: ${att_true:.2f}")

# ATC: Average effect among control
atc_true = (Y1[T==0] - Y0[T==0]).mean()
print(f"True ATC: ${atc_true:.2f}")

# ATE is weighted average of ATT and ATC
p_treated = T.mean()
ate_from_att_atc = p_treated * att_true + (1 - p_treated) * atc_true
print(f"ATE from ATT and ATC: ${ate_from_att_atc:.2f}")
print(f"Direct ATE estimate: ${true_ate:.2f}")
```

### Visualizing Potential Outcomes

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Distribution of potential outcomes
axes[0].hist(Y0, bins=30, alpha=0.5, label='Y(0) - No Treatment', density=True)
axes[0].hist(Y1, bins=30, alpha=0.5, label='Y(1) - Treatment', density=True)
axes[0].axvline(Y0.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean Y(0) = ${Y0.mean():.0f}')
axes[0].axvline(Y1.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean Y(1) = ${Y1.mean():.0f}')
axes[0].set_xlabel('Earnings')
axes[0].set_ylabel('Density')
axes[0].set_title('Distribution of Potential Outcomes')
axes[0].legend()

# Plot 2: Individual treatment effects
treatment_effects = Y1 - Y0
axes[1].hist(treatment_effects, bins=30, color='green', alpha=0.7, density=True)
axes[1].axvline(treatment_effects.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean ATE = ${treatment_effects.mean():.0f}')
axes[1].set_xlabel('Treatment Effect (Y(1) - Y(0))')
axes[1].set_ylabel('Density')
axes[1].set_title('Distribution of Individual Treatment Effects')
axes[1].legend()

plt.tight_layout()
plt.savefig('../figures/02-potential-outcomes.png', dpi=150, bbox_inches='tight')
plt.show()
```

### The Fundamental Problem in Action

```python
# Show what we observe vs what we need
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# What we observe
axes[0].scatter(range(n_total), Y, c=T, cmap='coolwarm', alpha=0.5, s=10)
axes[0].set_xlabel('Unit')
axes[0].set_ylabel('Observed Outcome')
axes[0].set_title('What We Observe\n(Red = Treated, Blue = Control)')
axes[0].axhline(y=Y[T==1].mean(), color='red', linestyle='--', label=f'Mean Treated = ${Y[T==1].mean():.0f}')
axes[0].axhline(y=Y[T==0].mean(), color='blue', linestyle='--', label=f'Mean Control = ${Y[T==0].mean():.0f}')
axes[0].legend()

# What we would need (potential outcomes)
for i in range(min(50, n_total)):  # Show first 50 units
    if T[i] == 1:
        axes[1].plot([i, i], [Y0[i], Y1[i]], 'r-', alpha=0.3, linewidth=1)
        axes[1].scatter(i, Y1[i], c='red', s=20, zorder=5)
        axes[1].scatter(i, Y0[i], c='red', s=10, marker='x', zorder=5)
    else:
        axes[1].plot([i, i], [Y0[i], Y1[i]], 'b-', alpha=0.3, linewidth=1)
        axes[1].scatter(i, Y0[i], c='blue', s=20, zorder=5)
        axes[1].scatter(i, Y1[i], c='blue', s=10, marker='x', zorder=5)

axes[1].set_xlabel('Unit')
axes[1].set_ylabel('Outcome')
axes[1].set_title('What We Would Need\n(Solid = Observed, X = Counterfactual)')

plt.tight_layout()
plt.savefig('../figures/02-fundamental-problem.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Reproducing the Lalonde Results

```python
# Simulate the Lalonde setup
np.random.seed(123)

# NSW participants (treated and control from experiment)
n_nsw = 445
nsw_income_0 = np.random.normal(5000, 3000, n_nsw)
nsw_income_1 = nsw_income_0 + np.random.normal(1800, 500, n_nsw)

# CPS comparison group (not randomized)
n_cps = 15000
cps_income_0 = np.random.normal(20000, 8000, n_cps)
cps_income_1 = cps_income_0  # No treatment effect in CPS (they didn't get training)

# Combine data
nsw_data = pd.DataFrame({
    'income': np.concatenate([nsw_income_1[:185], nsw_income_0[185:]]),
    'treated': np.concatenate([np.ones(185), np.zeros(n_nsw-185)]),
    'group': 'NSW'
})

cps_data = pd.DataFrame({
    'income': cps_income_0,
    'treated': np.zeros(n_cps),
    'group': 'CPS'
})

# Experimental estimate (using NSW data)
experimental_effect = nsw_data[nsw_data['treated']==1]['income'].mean() - \
                      nsw_data[nsw_data['treated']==0]['income'].mean()

print(f"Experimental estimate (NSW only): ${experimental_effect:.2f}")
print(f"True treatment effect: ${np.mean(nsw_income_1 - nsw_income_0):.2f}")

# Naive estimate comparing NSW treated to CPS control
naive_effect = nsw_data[nsw_data['treated']==1]['income'].mean() - \
               cps_data['income'].mean()

print(f"\nNaive estimate (NSW treated vs CPS control): ${naive_effect:.2f}")
print("(This is biased because CPS is not comparable to NSW)")
```

---

## 2.8 Common Mistakes

### Mistake 1: Confusing ATE with ATT

**The error**: Reporting the overall treatment effect when you should be reporting the effect for the treated.

**Why it matters**: The ATE might be small while the ATT is large (or vice versa). Policy decisions often depend on which quantity is relevant.

**How to avoid it**: Clearly state which estimand you're targeting and why.

### Mistake 2: Ignoring SUTVA Violations

**The error**: Assuming no interference when spillovers are likely.

**Examples**:
- Job training programs that affect local labor markets
- Vaccination programs with herd immunity effects
- Educational interventions that change classroom dynamics

**How to detect**: Think carefully about whether one unit's treatment could affect another unit's outcome.

### Mistake 3: Assuming Exchangeability Without Justification

**The error**: Claiming causal effects from observational data without establishing exchangeability.

**How to avoid it**: Draw a DAG, identify confounders, and conduct sensitivity analysis.

### Mistake 4: Confusing Consistency with No Multiple Versions of Treatment

**The error**: Assuming consistency holds when treatment is heterogeneous.

**Example**: If "job training" includes 100 different programs with different effects, consistency may not hold.

---

## 2.9 Discussion Questions

1. **The Baby Video Effect**: A study finds that babies who watch "Baby Einstein" videos have higher test scores. Is this causal? What potential outcomes framework would you use to evaluate this claim?

2. **Minimum Wage and Employment**: Card and Krueger (1994) used a natural experiment to study minimum wage effects. How would you apply the potential outcomes framework to this question?

3. **SUTVA in Practice**: Give three examples where SUTVA might be violated in a medical study. How would you handle these violations?

4. **Which Estimand Matters?**: A government program is being evaluated. Should we focus on ATE, ATT, or ATC? What are the policy implications of each?

5. **The Lalonde Puzzle**: Why did observational methods fail to replicate the experimental results? What does this tell us about the limitations of observational studies?

---

## 2.10 Interview Questions

### Question 1
**"What is the Fundamental Problem of Causal Inference? How do we address it?"**

**Model Answer**: The Fundamental Problem is that we can never observe both potential outcomes for the same unit. If someone receives treatment, we can't see what would have happened without treatment, and vice versa. This means individual causal effects are fundamentally unknowable. We address this by working with population-level quantities (ATE, ATT, ATC) and using methods like randomization, matching, or instrumental variables to ensure that treated and control groups are comparable. The key insight is that causal inference is fundamentally about missing data—we need assumptions (like exchangeability) to fill in the missing counterfactuals.

### Question 2
**"Explain SUTVA and give an example where it fails."**

**Model Answer**: SUTVA (Stable Unit Treatment Value Assumption) has two parts: (1) no interference—one unit's treatment doesn't affect another's outcome, and (2) treatment variation irrelevance—there's only one version of each treatment level. An example of failure is a vaccination program: if I get vaccinated, it reduces your infection risk through herd immunity, violating no interference. Another example is job training in a small town: if many people get trained, they might compete for the same jobs, affecting each other's outcomes.

### Question 3
**"What is the difference between ATE, ATT, and ATC? When would each be most relevant?"**

**Model Answer**: ATE is the average effect across the entire population—relevant for understanding the overall impact of a policy. ATT is the average effect among those who actually received treatment—relevant for evaluating programs for participants. ATC is the average effect among non-recipients—relevant for understanding what would happen if we expanded treatment. For example, in evaluating a job training program, ATT tells us the benefit for participants, while ATE tells us the average benefit if everyone participated.

### Question 4
**"What is exchangeability and why is it important?"**

**Model Answer**: Exchangeability means that treatment assignment is independent of potential outcomes, conditional on observed covariates. It's important because it allows us to estimate causal effects by comparing treated and control groups. In a randomized experiment, exchangeability holds by design. In observational studies, we need to measure and control for all confounders to achieve exchangeability. If exchangeability doesn't hold (unmeasured confounding), our causal estimates will be biased.

### Question 5
**"Design a study to evaluate the effect of a new drug on blood pressure."**

**Model Answer**: The gold standard is a randomized controlled trial: recruit patients, randomly assign to drug or placebo, measure blood pressure after treatment. Key design considerations: (1) sample size calculation based on expected effect size, (2) blinding (double-blind if possible), (3) placebo control, (4) adequate follow-up period, (5) intention-to-treat analysis, (6) handling of dropouts. If randomization isn't possible, consider propensity score matching, instrumental variables, or difference-in-differences using policy changes.

---

## 2.11 Knowledge Check

### Multiple Choice

1. **The Fundamental Problem of Causal Inference is:**
   - A) We can't measure outcomes accurately
   - B) We can never observe both potential outcomes for the same unit
   - C) Sample sizes are too small
   - D) Randomization is impossible

2. **Exchangeability means:**
   - A) Treatment and control groups have the same outcomes
   - B) Treatment assignment is independent of potential outcomes
   - C) There are no confounders
   - D) The treatment effect is the same for everyone

3. **SUTVA includes:**
   - A) No interference between units
   - B) No multiple versions of treatment
   - C) Both A and B
   - D) Neither A nor B

4. **The ATE can be written as:**
   - A) $E[Y(1)] - E[Y(0)]$
   - B) $E[Y | T=1] - E[Y | T=0]$ under exchangeability
   - C) Both A and B
   - D) Neither A nor B

5. **If exchangeability fails, the naive difference in means is:**
   - A) Unbiased
   - B) Biased due to confounding
   - C) Consistent
   - D) Efficient

### Short Answer

6. **Define potential outcomes and explain why they're central to causal inference.**

7. **Explain why randomized experiments allow us to estimate causal effects without conditioning on covariates.**

8. **What is the relationship between ATE, ATT, and ATC?**

9. **Give an example where positivity might fail in a medical study.**

10. **Explain the difference between consistency and exchangeability.**

### Critical Thinking

11. **A researcher claims to have estimated the causal effect of college education on earnings using observational data. What assumptions are they making? How could you evaluate whether these assumptions are reasonable?**

12. **The Lalonde study found that many observational methods failed to replicate experimental results. What does this tell us about the challenges of causal inference from observational data?**

---

## 2.12 Practical Checklist

Before making any causal claim using potential outcomes:

- [ ] Have I clearly defined the treatment and potential outcomes?
- [ ] Have I specified which estimand I'm targeting (ATE, ATT, ATC)?
- [ ] Have I assessed whether SUTVA is plausible?
- [ ] Have I identified and controlled for all confounders?
- [ ] Have I checked whether exchangeability is reasonable?
- [ ] Have I verified that positivity holds in my sample?
- [ ] Have I considered the possibility of multiple versions of treatment?
- [ ] Have I conducted sensitivity analysis for unmeasured confounding?
- [ ] Have I been honest about what my analysis can and cannot conclude?

---

## Watch and Connect

```{admonition} Recommended lecture
:class: tip

[**Potential outcomes and a complete estimation example**](https://www.youtube.com/watch?v=q8x9aetyok0&list=PLoazKTcS0Rzb6bb9L508cyJ1z-U9iWkA0&index=8)

A compact bridge from counterfactual notation to an estimable causal effect.

**Active-viewing prompt:** Pause when ignorability appears and state what would make it implausible.
```

---

## 2.13 Summary

In this lesson, we introduced the Potential Outcomes Framework:

1. **Potential Outcomes**: For each unit, define $Y(1)$ (outcome under treatment) and $Y(0)$ (outcome under control).

2. **Fundamental Problem**: We can only observe one potential outcome per unit, making individual causal effects unknowable.

3. **Average Treatment Effects**: We work with population quantities (ATE, ATT, ATC) instead of individual effects.

4. **Key Assumptions**:
   - SUTVA: No interference, treatment variation irrelevance
   - Exchangeability: Treatment independent of potential outcomes (conditional on covariates)
   - Consistency: Observed outcome equals the potential outcome for the observed treatment
   - Positivity: Every unit has a positive probability of receiving either treatment

5. **Randomized Experiments**: Provide a benchmark because exchangeability holds by design.

6. **Observational Studies**: Require additional assumptions and methods to estimate causal effects.

In the next lesson, we'll learn about Directed Acyclic Graphs (DAGs), which provide a graphical framework for encoding causal assumptions.

---

## 2.14 Further Reading

### Classic Works
- Rubin, D.B. (1974). "Estimating Causal Effects of Treatments in Randomized and Nonrandomized Studies." *Journal of Educational Psychology*.
- Rubin, D.B. (1978). "Bayesian Inference for Causal Effects: The Role of Randomization." *The Annals of Statistics*.
- Holland, P.W. (1986). "Statistics and Causal Inference." *Journal of the American Statistical Association*.

### Modern Textbooks
- Imbens, G.W. & Rubin, D.B. (2015). *Causal Inference for Statistics, Social, and Biomedical Sciences*. Cambridge University Press.
- Rubin, D.B. (2015). *Causal Inference for Statistics, Social, and Biomedical Sciences*. Cambridge University Press.

### The Lalonde Study
- Lalonde, R.J. (1986). "Evaluating the Econometric Evaluations of Training Programs with Experimental Data." *American Economic Review*.
- Dehejia, R.H. & Wahba, S. (1999). "Causal Effects in Nonexperimental Studies: Reevaluating the Training Evaluation Program." *Journal of the American Statistical Association*.


---

## Worked Examples

### Example 1: Calculating the ATE from Potential Outcomes

If we knew everyone's potential outcomes Y(0) and Y(1), the ATE is simply the average difference.

```python
import numpy as np
import pandas as pd

np.random.seed(42)
n = 1000
treatment = np.random.binomial(1, 0.5, n)
y0 = np.random.normal(0, 1, n)
y1 = y0 + 2.0 + np.random.normal(0, 0.5, n)

ate = np.mean(y1 - y0)
print(f"True ATE: {ate:.3f}")
print(f"Observed difference (biased): {np.mean(y1[treatment==1]) - np.mean(y0[treatment==0]):.3f}")
```

### Example 2: The Fundamental Problem of Causal Inference

We never observe both Y(0) and Y(1) for the same individual. This is why causal inference is hard.

```python
import numpy as np

np.random.seed(42)
n = 10
# True potential outcomes
y0_true = np.random.normal(5, 1, n)
y1_true = y0_true + 2.0
# Treatment assignment
treatment = np.random.binomial(1, 0.5, n)
# We only observe one potential outcome per person
observed = np.where(treatment, y1_true, y0_true)

print("Person | Y(0) | Y(1) | Treated | Observed")
for i in range(n):
    print(f"  {i+1:4d} | {y0_true[i]:.2f} | {y1_true[i]:.2f} |    {treatment[i]}    | {observed[i]:.2f}")
print(f"\nTrue ATE: {np.mean(y1_true - y0_true):.3f}")
print(f"Naive estimate: {np.mean(observed[treatment==1]) - np.mean(observed[treatment==0]):.3f}")
```


---

## Exercises

### Exercise 1

Define SUTVA (Stable Unit Treatment Value Assumption) and explain why it matters for causal inference.

### Exercise 2

Create a simulation where you calculate both the true ATE and the naive estimate, showing the bias from non-random treatment assignment.

### Exercise 3

Explain the difference between the Average Treatment Effect (ATE) and the Average Treatment Effect on the Treated (ATT).
