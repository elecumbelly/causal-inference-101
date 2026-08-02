---
title: "Lesson 5: Randomized Controlled Trials"
subtitle: "From First Principles to Modern Causal Inference"
---

# Lesson 5: Randomized Controlled Trials

## Opening Story: The Oregon Medicaid Lottery

In 2008, the state of Oregon faced a problem. It wanted to expand its Medicaid program to cover low-income adults, but the budget only allowed for a limited expansion. The state decided to use a lottery—literally a random drawing—to decide who would get coverage.

For researchers, this was a golden opportunity. Oregon's Medicaid expansion wasn't designed as a research study, but the random lottery created something close to a randomized controlled trial: a natural experiment where treatment (Medicaid coverage) was assigned by chance.

The Oregon Health Insurance Experiment, analyzed by Finkelstein et al. (2012) and Baicker et al. (2013), became one of the most important studies in health economics. It showed that Medicaid coverage increased healthcare utilization, reduced financial strain, and improved self-reported health—but the effects on measured health outcomes like blood pressure were mixed.

This lesson is about why randomized experiments are the gold standard for causal inference, how they work, and what can go wrong when they don't.

---

## Learning Objectives

By the end of this lesson, you should be able to:

1. Explain why randomization establishes exchangeability
2. Distinguish between ITT and treatment-on-the-treated estimates
3. Identify and handle compliance issues
4. Recognize threats to validity including attrition
5. Discuss external validity limitations
6. Design and analyze a randomized experiment in Python

---

```{figure} ../figures/instructional/rct-flow.svg
:name: lesson-05-rct-flow
:alt: Assignment and treatment receipt separate when participants do not comply.
:width: 100%

Assignment and treatment receipt separate when participants do not comply.
```

---

## 5.1 Why Randomization Works

### The Key Insight

In a randomized experiment:
- Treatment assignment $T$ is independent of potential outcomes $(Y(0), Y(1))$
- This means: $(Y(0), Y(1)) \perp\!\!\!\perp T$
- Therefore: $E[Y(0) | T = 1] = E[Y(0) | T = 0]$

This is exchangeability **by design**—we don't need to condition on any covariates.

### The Math

Under randomization:

$$ATE = E[Y | T = 1] - E[Y | T = 0]$$

The difference in observed outcomes equals the causal effect because randomization ensures the groups are comparable.

### Why This Is Revolutionary

Before randomized experiments, researchers relied on observational comparisons. The problem was always: "How do I know the groups are comparable?" Randomization answers this question definitively—by making treatment assignment unrelated to all factors (observed and unobserved) that affect the outcome.

---

## 5.2 Compliance Issues

### The Problem

In practice, not everyone complies with their assigned treatment:
- Some people assigned to treatment don't receive it (non-compliance)
- Some people assigned to control receive treatment anyway (contamination)

### Intention-to-Treat (ITT)

The ITT estimate compares outcomes based on **assignment**, regardless of actual treatment:

$$ITT = E[Y | Z = 1] - E[Y | Z = 0]$$

where $Z$ is the random assignment.

**Advantages**:
- Preserves randomization
- Easy to estimate
- Policy-relevant (tells you the effect of being assigned treatment)

**Disadvantages**:
- Underestimates the effect of actual treatment (diluted by non-compliance)

### Treatment-on-the-Treated (TOT)

The TOT estimate measures the effect of actual treatment among those who received it:

$$TOT = E[Y(1) - Y(0) | T = 1]$$

**Estimation**: Use instrumental variables where $Z$ (assignment) is the instrument for $T$ (actual treatment).

---

## 5.3 Attrition

### The Problem

Attrition occurs when participants drop out of the study. If attrition is related to treatment or outcomes, it can bias results.

### Types of Attrition

- **Missing Completely at Random (MCAR)**: Attrition is unrelated to anything
- **Missing at Random (MAR)**: Attrition depends on observed variables only
- **Missing Not at Random (MNAR)**: Attrition depends on unobserved outcomes

### Handling Attrition

1. **ITT analysis**: Include all randomized participants
2. **Bounds**: Calculate bounds under worst-case assumptions (Manski bounds)
3. **Inverse probability weighting**: Weight by probability of remaining in study
4. **Sensitivity analysis**: Assess how sensitive results are to attrition assumptions

---

## 5.4 External Validity

### The Question

Even if the experiment has high internal validity (we can trust the causal estimate), does it generalize to other settings, populations, or time periods?

### Threats to External Validity

1. **Population effects**: The sample may not represent the broader population
2. **Treatment effects**: The treatment in the experiment may differ from real-world implementation
3. **Setting effects**: The experimental setting may differ from the real world
4. **Time effects**: Results may not hold in different time periods

### Evaluating External Validity

- Compare sample characteristics to the target population
- Assess whether the treatment is implemented similarly
- Consider whether the context is representative
- Look for replication in different settings

---

## 5.5 Python Workshop: Simulating an RCT

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression

np.random.seed(42)

# Simulate a job training experiment
n = 1000

# Baseline characteristics
age = np.random.normal(35, 10, n)
education = np.random.normal(12, 2, n)
baseline_income = np.random.normal(20000, 8000, n)

# Random assignment
Z = np.random.binomial(1, 0.5, n)

# Actual treatment (with non-compliance)
# Some people assigned to treatment don't show up
compliance_prob = 0.7 + 0.01 * education  # More educated more likely to comply
T = Z * np.random.binomial(1, compliance_prob) + (1-Z) * np.random.binomial(1, 0.05)

# Potential outcomes
Y0 = 5000 + 200 * education + 100 * age + 0.1 * baseline_income + np.random.normal(0, 3000, n)
effect = 3000 + 10 * education  # Heterogeneous effect
Y1 = Y0 + effect

# Observed outcome
Y = Y1 * T + Y0 * (1 - T)

# Create dataframe
df = pd.DataFrame({
    'age': age, 'education': education, 'baseline_income': baseline_income,
    'Z': Z, 'T': T, 'Y': Y, 'Y0': Y0, 'Y1': Y1
})

# ITT estimate
itt = df[df['Z']==1]['Y'].mean() - df[df['Z']==0]['Y'].mean()

# True ATE
true_ate = (df['Y1'] - df['Y0']).mean()

# Naive TOT (biased by selection)
tot_naive = df[df['T']==1]['Y'].mean() - df[df['T']==0]['Y'].mean()

print(f"True ATE: ${true_ate:.2f}")
print(f"ITT estimate: ${itt:.2f}")
print(f"Naive TOT: ${tot_naive:.2f}")

# Compliance rate
compliance = (T[Z==1].sum() / Z.sum())
print(f"\nCompliance rate: {compliance:.2%}")
print(f"ITT / Compliance ≈ LATE: ${itt/compliance:.2f}")
```

### Checking Balance

```python
# Check that randomization created balanced groups
balance_vars = ['age', 'education', 'baseline_income']

print("Balance Check:")
print("-" * 50)
for var in balance_vars:
    treated_mean = df[df['Z']==1][var].mean()
    control_mean = df[df['Z']==0][var].mean()
    std_diff = (treated_mean - control_mean) / df[var].std()
    print(f"{var:20s}: Treated={treated_mean:.2f}, Control={control_mean:.2f}, Std Diff={std_diff:.3f}")

# Visualize balance
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, var in enumerate(balance_vars):
    axes[i].hist(df[df['Z']==0][var], bins=30, alpha=0.5, label='Control', density=True)
    axes[i].hist(df[df['Z']==1][var], bins=30, alpha=0.5, label='Treated', density=True)
    axes[i].set_title(f'{var} Distribution')
    axes[i].legend()
plt.tight_layout()
plt.savefig('../figures/05-rct-balance.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 5.6 Case Study: Oregon Medicaid

### The Setup

- 89,824 people applied for Medicaid expansion
- 30,000 were selected by lottery
- Lottery winners had a chance to enroll in Medicaid

### Key Findings

- Medicaid increased healthcare utilization (30% increase in ER visits initially)
- Reduced financial strain (less catastrophic medical expenditures)
- Improved self-reported health
- No significant effects on measured health outcomes (blood pressure, cholesterol)
- Significant reduction in depression (25% reduction)

### Why Results Mattered

The Oregon experiment showed that:
1. Health insurance affects healthcare utilization
2. Financial protection is a key benefit
3. Self-reported health may respond differently than clinical measures
4. Depression is strongly affected by financial security

---

## Interference, Spillovers, and Cluster Randomization

Standard analyses assume one unit's treatment does not change another unit's outcome. That fails for vaccines, social networks, classrooms and geographic policies. If treated students share materials with control students, for example, the control condition is partly treated.

Three design responses are common:

1. **Cluster randomization** assigns intact groups such as schools or villages.
2. **Saturation designs** randomize both clusters and the proportion treated within clusters.
3. **Exposure mappings** redefine treatment to include neighbours' assignments.

Cluster assignment reduces contamination but also reduces effective sample size. Power and standard errors must reflect the number and correlation of clusters—not merely the number of individuals.

## Target-Trial Emulation

When an experiment cannot be run, define the hypothetical trial before analysing observational data. Specify its eligibility criteria, treatment strategies, assignment procedure, time zero, follow-up, outcome, causal contrast and analysis plan. Then map each element to the available data.

This discipline prevents common errors such as immortal-time bias, selecting people using future information, or starting follow-up at different moments for treated and untreated groups. Emulation does not remove confounding, but it makes the causal question and design failures visible.

---

## 5.7 Common Mistakes

1. **Ignoring non-compliance**: Always report ITT and consider TOT
2. **Not checking balance**: Verify that randomization worked
3. **Ignoring attrition**: Track all participants, analyze missing data
4. **Overclaiming external validity**: Experiments have limited generalizability
5. **P-hacking**: Don't search for statistically significant results

---

## 5.8 Discussion Questions

1. **The Oregon Experiment**: Oregon's Medicaid lottery wasn't a perfect experiment. What threats to internal validity exist? How did researchers address them?

2. **Ethical Dilemmas**: Is it ethical to randomize access to beneficial treatments? When is randomization justified?

3. **Sample Selection**: The Oregon experiment studied people who applied for Medicaid. Does this limit external validity?

4. **Compliance**: If only 70% of those assigned to treatment actually receive it, what does the ITT estimate? How can we estimate the effect of actual treatment?

5. **Replication**: How would you design a follow-up study to test whether Oregon's results generalize to other states?

---

## 5.9 Knowledge Check

### Multiple Choice

1. **The primary advantage of randomization is:**
   - A) Larger sample sizes
   - B) Exchangeability by design
   - C) Lower cost
   - D) Faster results

2. **ITT estimates the effect of:**
   - A) Actual treatment
   - B) Assignment to treatment
   - C) Both A and B
   - D) Neither A nor B

3. **Non-compliance causes ITT to:**
   - A) Overestimate the true effect
   - B) Underestimate the true effect
   - C) Not change the estimate
   - D) Make the estimate inconsistent

4. **Attrition is most problematic when it is:**
   - A) Random
   - B) Related to treatment assignment
   - C) Related to outcomes
   - D) Both B and C

5. **External validity asks:**
   - A) Are the results statistically significant?
   - B) Do the results generalize to other settings?
   - C) Are the confounders controlled?
   - D) Is the sample size large enough?

### Short Answer

6. **Explain why randomization establishes exchangeability without conditioning on covariates.**

7. **What is the difference between ITT and TOT? When would you prefer one over the other?**

8. **How can you assess whether attrition is likely to bias your results?**

9. **Describe three threats to external validity in a job training experiment.**

10. **Why is compliance never 100% in real experiments? What are the consequences?**

---

## Watch and Connect

```{admonition} Recommended lecture
:class: tip

[**Randomized experiments and graphical identification**](https://www.youtube.com/watch?v=z91LnTDyhtI&list=PLoazKTcS0Rzb6bb9L508cyJ1z-U9iWkA0&index=37)

Explains what randomization identifies—and what noncompliance and missingness can still break.

**Active-viewing prompt:** Separate the effect of assignment from the effect of treatment received.
```

---

## 5.10 Summary

In this lesson, we learned:

1. **Randomization establishes exchangeability** by design, eliminating confounding
2. **ITT estimates the effect of assignment**, which is conservative but valid
3. **TOT estimates the effect of actual treatment** but requires additional assumptions
4. **Non-compliance, attrition, and external validity** are practical challenges
5. **The Oregon Medicaid experiment** demonstrated both the power and limitations of RCTs

---

## 5.11 Further Reading

- Finkelstein, A. et al. (2012). "The Oregon Health Insurance Experiment." *American Economic Review*.
- Baicker, K. et al. (2013). "The Oregon Experiment—Effects of Medicaid on Clinical Outcomes." *New England Journal of Medicine*.
- Imbens, G.W. & Rubin, D.B. (1994). "Estimating the Effect of Treatments in Randomized and Nonrandomized Studies." *Psychological Bulletin*.


---

## Worked Examples

### Example 1: Medicine — Drug Efficacy Trial

A pharmaceutical company tests a new blood pressure medication. They recruit 500 patients, randomly assign 250 to treatment and 250 to placebo, and measure blood pressure reduction after 12 weeks.

**Why randomization works here:**
- Age, diet, genetics, and severity are balanced across groups on average
- The only systematic difference is the drug
- Any difference in outcomes can be attributed to the treatment

**Potential problems:**
- **Attrition**: Patients who feel worse may drop out, biasing results
- **Non-compliance**: Some patients stop taking the medication
- **Contamination**: Control group patients may access the drug elsewhere

### Example 2: Economics — Job Training Program

A government evaluates a job training program. They randomly select 1,000 applicants from 10,000, offer training to 500, and compare earnings after 2 years.

**Challenge**: Even with random assignment, the program has **non-compliance**. Some offered training don't attend; some not offered find training elsewhere. This creates a **per-protocol** vs **intention-to-treat** distinction.

| Analysis | Compares | Estimates |
|----------|----------|-----------|
| Intention-to-treat (ITT) | All offered vs all not offered | Effect of being *offered* training |
| Per-protocol | Those who actually trained vs those who didn't | Effect of *receiving* training (biased!) |
| LATE via IV | Uses offer as instrument | Effect for *compliers* |

### Example 3: Technology — A/B Testing

A social media platform tests a new feed algorithm. They randomly show 10% of users the new algorithm and 90% the old one, measuring engagement over 2 weeks.

**Scale advantage**: With millions of users, even tiny effects are detectable. A 0.1% increase in engagement translates to millions of additional minutes of watch time.

**Pitfalls:**
- **Multiple testing**: Testing 50 metrics at p < 0.05 guarantees ~2 false positives
- **Network effects**: Users interact with each other, violating the stable unit treatment value assumption (SUTVA)
- **Novelty effect**: Users may engage more initially just because something is different

### Example 4: Policy — Education Class Size Reduction

A state randomly assigns 100 schools to reduce class sizes from 30 to 15 students, tracking test scores for 3 years.

**Logistical challenges:**
- Schools may reallocate teachers, confounding the intervention
- Parents in treatment schools may respond differently (Hawthorne effect)
- Outcomes take years to manifest
- Cost per student is enormous

---

## Diagnostics: Common Pitfalls

### Checklist for RCT Quality

| Check | Question | Red Flag |
|-------|----------|----------|
| **Randomization** | Was allocation truly random? | Sequential assignment, date-of-birth assignment |
| **Allocation concealment** | Could investigators predict assignment? | Open randomization tables, alternating assignment |
| **Blinding** | Were participants and assessors blinded? | Open-label when blinding is possible |
| **Sample size** | Was power analysis conducted? | Post-hoc sample size justification |
| **Attrition** | Is dropout rate similar across groups? | >20% differential attrition |
| **Intention-to-treat** | Was primary analysis ITT? | Per-protocol as primary analysis |
| **Pre-registration** | Was the trial pre-registered? | Unregistered primary outcomes |

### The CONSORT Flow Diagram

Every well-reported RCT should include a CONSORT flow diagram showing:
1. How many participants were assessed for eligibility
2. How many were excluded and why
3. How many were randomized to each arm
4. How many received the intended intervention
5. How many were lost to follow-up
6. How many were included in the final analysis

This transparency allows readers to assess the internal validity of the trial.

---

## Interpretation Workshop

### Reading an RCT Results Table

When evaluating an RCT results table, check:

1. **Baseline balance**: Are treatment and control groups similar on pre-treatment characteristics?
2. **Effect size**: Is the observed difference clinically or practically meaningful?
3. **Precision**: Is the confidence interval narrow enough for decision-making?
4. **Absolute vs relative effects**: A 50% relative reduction might mean going from 2% to 1% risk

### Common Misinterpretations

- "No statistically significant difference" does NOT mean "no effect" — the study may be underpowered
- "Statistically significant" does NOT mean "large or important" — with enough subjects, trivial effects become significant
- ITT estimates the effect of being *offered* treatment, not the effect of *receiving* it

---

## Practical Application

### Designing Your Own RCT

**Step 1: Define the estimand**
- What specific causal effect do you want to estimate?
- For what population?
- Under what conditions?

**Step 2: Choose the unit of randomization**
- Individual: Standard for clinical trials
- Cluster: When treatment is delivered to groups (schools, hospitals)
- Stepped-wedge: When rollout is gradual and universal coverage is the goal

**Step 3: Calculate sample size**
- What effect size matters for decision-making?
- What Type I error rate (alpha) and power (1-beta) do you need?
- Use power analysis software or formulas

**Step 4: Implement with fidelity**
- Monitor adherence throughout the trial
- Document any deviations from the protocol
- Track adverse events

---

## Limitations

- **Ethical constraints**: Cannot randomize harmful exposures
- **Practical constraints**: RCTs are expensive and time-consuming
- **Generalizability**: Trial populations may not represent real-world patients
- **Compliance**: Real-world adherence is lower than in controlled settings
- **External validity**: Results may not transport to different settings or populations

---

## Exercises

1. **Design exercise**: A city wants to know if a new bus route reduces commute times. Design an RCT. What are the challenges? How would you handle spillover effects?

2. **Critical appraisal**: Find a published RCT and evaluate it against the CONSORT checklist. What is missing? Would you trust the results?

3. **Power analysis**: You want to detect a 2-point difference in SAT scores (SD = 100). How many students do you need per group at 80% power and alpha = 0.05? Is this feasible?

4. **Ethics exercise**: A drug shows promise in phase II trials. Phase III requires a placebo control, but patients in the control group may die. Discuss the ethical tensions and how you would resolve them.

---

## Projects

### Project 1: Simulate an RCT
Write Python code that:
1. Generates a population with correlated covariates
2. Randomly assigns units to treatment/control
3. Implements non-compliance
4. Estimates ITT and treatment-on-treated effects
5. Compares results to the true causal effect

### Project 2: Analyze a Real RCT
Find a published RCT in your field of interest. Using the published data or summary statistics:
1. Verify the reported analysis
2. Conduct subgroup analyses
3. Assess robustness to different assumptions
4. Write a critical appraisal following CONSORT guidelines
