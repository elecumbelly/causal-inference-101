---
title: "Lesson 1: Why Causality Matters"
subtitle: "From First Principles to Modern Causal Inference"
---

# Lesson 1: Why Causality Matters

## Opening Story

In 1948, the British Medical Research Council conducted what many consider the first modern randomized controlled trial. They wanted to know whether streptomycin could cure tuberculosis.

The problem was obvious: patients with tuberculosis were dying, and doctors had no effective treatment. Observational evidence suggested streptomycin might help—but how could they be sure?

The researchers faced a fundamental dilemma. If they gave streptomycin to all patients, they wouldn't know whether improvement was due to the drug or to natural recovery. If they compared treated patients to untreated patients, they couldn't be sure the groups were comparable.

Their solution was elegant: **randomly assign** some patients to receive streptomycin and others to receive the standard bed rest. If the groups were large enough and randomization was properly implemented, the only systematic difference between them would be the treatment.

The results were clear. Streptomycin patients had dramatically better outcomes. This trial, and the methodology it pioneered, has saved millions of lives since.

But here's the question that should haunt you: **Why did randomization work?** What was it about randomly assigning patients that allowed the researchers to draw causal conclusions?

The answer to that question is the heart of this course.

---

## Learning Objectives

By the end of this lesson, you should be able to:

1. Distinguish between association, intervention, and counterfactual reasoning
2. Explain why correlation does not imply causation
3. Identify the three fundamental problems of causal inference
4. Describe Pearl's Ladder of Causation
5. Recognize common sources of confounding
6. Understand the concept of counterfactuals
7. Explain why decisions require causal, not merely predictive, knowledge

---

## 1.1 Why Humans Seek Causes

### The Story-Telling Animal

Humans are, at our core, causal creatures. We don't just observe patterns—we demand explanations for them.

When a child asks "why is the sky blue?", they're not satisfied with "it just is." When a detective arrives at a crime scene, they don't just catalog evidence—they construct a narrative of what happened and why. When a doctor sees a patient with symptoms, they don't just treat the symptoms—they diagnose the underlying cause.

This causal drive is deeply rooted in our evolutionary history. Our ancestors who could correctly identify why a rustling in the bushes meant danger (predator) versus safety (wind) had survival advantages over those who couldn't.

### From Patterns to Causes

Consider these three statements:

1. "People who carry lighters are more likely to develop lung cancer."
2. "Students who attend test preparation courses score higher on standardized tests."
3. "Countries with higher chocolate consumption produce more Nobel Prize winners."

Each describes a **correlation**—a statistical association between two variables. But our causal minds immediately jump to questions:

- Does carrying lighters *cause* cancer?
- Does test prep *cause* higher scores?
- Does chocolate *cause* Nobel prizes?

The answers reveal something fundamental about the difference between observing patterns and understanding causes.

---

## 1.2 Description, Prediction, and Causation

### Three Different Questions

Data science and statistics traditionally focus on three types of questions:

| Question Type | What It Answers | Example |
|---------------|-----------------|---------|
| **Descriptive** | What happened? | "Smokers have a 15% higher rate of lung cancer" |
| **Predictive** | What will happen? | "Given these symptoms, there's an 80% chance of disease X" |
| **Causal** | What would happen if we intervened? | "If we ban smoking, lung cancer rates will fall by 12%" |

These are fundamentally different questions that require different tools.

**Descriptive statistics** summarize what we observe. They're useful for understanding patterns but tell us nothing about why those patterns exist.

**Prediction** uses observed patterns to forecast future outcomes. A model that predicts heart attacks based on cholesterol levels, blood pressure, and age can be useful for screening—but it doesn't tell us whether *changing* cholesterol levels will *change* heart attack risk.

**Causal inference** asks what would happen if we *intervened*—if we changed something about the world. This is the hardest question to answer, but it's the question that matters most for decision-making.

### Why Prediction Isn't Enough

Consider a hospital that uses a predictive model to identify patients at risk of readmission. The model might work well—predicting which patients will return within 30 days with 85% accuracy.

But what if the hospital wants to *reduce* readmissions? Now prediction isn't enough. They need to know which factors *cause* readmissions, because only causes can be targeted for intervention.

A predictive model might find that patients who live far from the hospital are more likely to be readmitted. But is distance a *cause*? Or is distance correlated with other factors—poverty, lack of transportation, limited access to follow-up care—that actually cause readmissions?

If the hospital builds a satellite clinic near the distant patients' homes, will readmissions drop? Only if distance is a *cause*, not merely a correlate.

This distinction matters enormously in practice:

- **Marketing**: Should we send discount coupons to customers predicted to churn, or to customers whose churn would be *caused* by not receiving coupons?
- **Medicine**: Should we prescribe expensive drugs to patients predicted to have poor outcomes, or to patients whose outcomes would *improve* with the drug?
- **Policy**: Should we invest in programs predicted to reduce crime, or programs that would *cause* crime to reduce if implemented?

---

## 1.3 Scientific Reasoning

### The Scientific Method as Causal Reasoning

The scientific method is fundamentally a framework for causal reasoning:

1. **Observation**: Notice a pattern ("smokers get more lung cancer")
2. **Hypothesis**: Propose a causal explanation ("smoking causes lung cancer")
3. **Prediction**: Derive testable predictions ("if we reduce smoking, cancer rates will fall")
4. **Experiment**: Test the predictions under controlled conditions
5. **Conclusion**: Accept, reject, or revise the hypothesis

The key insight is that experiments are the gold standard for causal inference because they allow us to *manipulate* potential causes and observe the effects.

But experiments aren't always possible. We can't randomly assign people to smoke for 30 years. We can't randomly assign countries to different economic policies. We can't randomly assign children to different family structures.

In these situations, we must rely on **observational data**—data where the treatment assignment was not controlled by the researcher. And observational data is full of pitfalls.

### The Problem of Confounded Observation

When we observe that smokers have higher lung cancer rates, we might be tempted to conclude that smoking causes cancer. But what if there's a third factor—say, genetics—that causes both smoking behavior and cancer susceptibility?

```{figure} ../figures/01-confounding.png
:name: confounding-example
:alt: Confounding diagram
:width: 300px

Confounding: A common cause creates a spurious association.
```

In this diagram, genetics causes both smoking and cancer. Smoking and cancer are associated, but the association doesn't necessarily mean smoking *causes* cancer.

This is the fundamental problem of observational causal inference: **we can't tell from observation alone whether an association is causal**.

---

## 1.4 Decisions Require Interventions

### The Decision-Maker's Dilemma

Every decision is implicitly a causal question. When you decide to:

- Take a medication → You're asking "Will this medication *cause* my symptoms to improve?"
- Invest in education → You're asking "Will this investment *cause* better outcomes?"
- Change a business strategy → You're asking "Will this change *cause* increased revenue?"

The philosopher James Woodward put it this way: causal knowledge is knowledge that allows us to *manipulate* outcomes through interventions.

### The Ladder of Causation

Judea Pearl formalized this intuition with his "Ladder of Causation":

```{figure} ../figures/ladder-of-causation.png
:name: ladder
:alt: Pearl's Ladder of Causation
:width: 400px

Pearl's Ladder of Causation
```

**Level 1: Association (Seeing)**
- Questions: "What is the probability of Y given X?"
- Example: "What is the probability of lung cancer given that someone smokes?"
- Method: Observe correlations in data

**Level 2: Intervention (Doing)**
- Questions: "What happens to Y if I do X?"
- Example: "What happens to lung cancer rates if we ban smoking?"
- Method: Randomized experiments, causal inference

**Level 3: Counterfactuals (Imagining)**
- Questions: "What would have happened to Y if X had been different?"
- Example: "Would this patient have survived if they had received the drug?"
- Method: Counterfactual reasoning, structural causal models

The crucial insight: **you cannot answer questions at a higher level using only data from a lower level.** No amount of observational data (Level 1) can, by itself, answer interventional questions (Level 2) or counterfactual questions (Level 3).

---

## 1.5 Correlation vs. Causation

### The Mantra

"Correlation does not imply causation" is perhaps the most important sentence in statistics. It's also the most frequently ignored.

### Why Correlation Doesn't Imply Causation

There are five fundamental reasons why two variables might be correlated without one causing the other:

1. **Common cause (confounding)**: A third variable causes both
   - Ice cream sales and drowning deaths are correlated because heat causes both
   - The correlation doesn't mean ice cream causes drowning

2. **Reverse causation**: The assumed cause and effect are reversed
   - Hospital visits and death rates are correlated
   - But hospitals don't cause death—sick people go to hospitals

3. **Selection bias**: The way we sample data creates a spurious association
   - Among professional basketball players, height and shooting ability are negatively correlated
   - But this is because short players who couldn't shoot were eliminated from the sample

4. **Coincidence**: Sometimes variables are correlated by chance
   - The number of Nicolas Cage movies correlates with swimming pool drownings
   - This is pure coincidence

5. **Mediation**: The correlation exists because one variable causes a third variable, which causes the second
   - Studying and good grades are correlated
   - But studying causes learning, and learning causes good grades

### The Ice Cream and Drowning Fallacy

Let's examine the classic ice cream example in detail:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Simulate data
np.random.seed(42)
n = 1000

# Temperature affects both ice cream sales and drowning
temperature = np.random.normal(25, 8, n)

# Ice cream sales increase with temperature
ice_cream_sales = 50 + 2 * temperature + np.random.normal(0, 10, n)

# Drowning deaths increase with temperature (more swimming)
drowning_deaths = 0.5 + 0.1 * temperature + np.random.normal(0, 2, n)

# Create dataframe
df = pd.DataFrame({
    'temperature': temperature,
    'ice_cream_sales': ice_cream_sales,
    'drowning_deaths': drowning_deaths
})

# Plot the correlation
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(df['ice_cream_sales'], df['drowning_deaths'], alpha=0.5)
axes[0].set_xlabel('Ice Cream Sales')
axes[0].set_ylabel('Drowning Deaths')
axes[0].set_title('Correlation: r = {:.2f}'.format(
    df['ice_cream_sales'].corr(df['drowning_deaths'])
))

axes[1].scatter(df['temperature'], df['ice_cream_sales'], alpha=0.5, label='Ice Cream')
axes[1].scatter(df['temperature'], df['drowning_deaths'], alpha=0.5, label='Drowning')
axes[1].set_xlabel('Temperature')
axes[1].set_ylabel('Count')
axes[1].set_title('Common Cause: Temperature')
axes[1].legend()

plt.tight_layout()
plt.savefig('../figures/01-ice-cream-drowning.png', dpi=150)
plt.show()
```

The correlation between ice cream sales and drowning deaths is real—it's just not causal. Temperature is the common cause of both.

---

## 1.6 Confounding

### The Definition

**Confounding** occurs when a variable (the confounder) is a common cause of both the treatment and the outcome, creating a non-causal association between them.

### The Confounding Triangle

```{figure} ../figures/01-confounding-triangle.png
:name: confounding-triangle
:alt: The confounding triangle
:width: 200px

The confounding triangle: Z causes both X and Y.
```

If we want to know whether X causes Y, but Z causes both X and Y, then the association between X and Y is "confounded" by Z.

### Examples of Confounding

| Treatment | Outcome | Confounder |
|-----------|---------|------------|
| Coffee | Heart disease | Smoking (smokers drink more coffee) |
| Hormone replacement therapy | Heart disease | Socioeconomic status (wealthier women more likely to take HRT) |
| Exercise | Longevity | Genetics (healthy people exercise more) |
| Aspirin | Heart attack prevention | Health-conscious behavior |

### The Bias

If we ignore confounding, we get biased estimates of causal effects. The bias is the difference between what we estimate and the true causal effect.

**Mathematically**: If the true causal effect is $\tau$, but we estimate the association $\delta$, the bias is:
$$\text{Bias} = \delta - \tau$$

In the presence of confounding, this bias can be:
- **Positive**: We overestimate the causal effect
- **Negative**: We underestimate the causal effect
- **Reversed**: We get the sign wrong entirely

---

## 1.7 Reverse Causation

### The Direction Problem

Reverse causation occurs when we mistake the direction of causation. We assume X causes Y, but actually Y causes X.

### Classic Examples

**Hospital visits and death rates**: People who visit hospitals more frequently have higher death rates. Does hospital visits cause death? Of course not—sick people visit hospitals.

**Firefighters and fire damage**: More firefighters at a fire is associated with more damage. But larger fires attract more firefighters, not the other way around.

**GPA and happiness**: Happy students tend to have higher GPAs. But is happiness causing good grades, or are good grades causing happiness (or both)?

### Detection

Reverse causation is difficult to detect from observational data alone. The key is to think carefully about:
1. **Temporal ordering**: Does the cause precede the effect?
2. **Mechanism**: Is there a plausible causal pathway?
3. **Instrumental variables**: Can we find a variable that affects the supposed cause but not the supposed effect?

---

## 1.8 Selection Bias

### The Sampling Problem

Selection bias occurs when the way we select our sample creates a distorted picture of the true relationship between variables.

### Berkson's Paradox

One of the most counterintuitive forms of selection bias is Berkson's paradox, where conditioning on a common effect creates a spurious association between its causes.

**Example**: Among patients in a hospital, having disease A and having disease B might appear negatively correlated, even if they're independent in the general population. This is because only people with at least one disease end up in the hospital.

```python
# Berkson's paradox simulation
np.random.seed(42)
n = 10000

# Disease A and B are independent
disease_a = np.random.binomial(1, 0.1, n)
disease_b = np.random.binomial(1, 0.1, n)

# You go to hospital if you have either disease
in_hospital = (disease_a == 1) | (disease_b == 1)

# Among hospital patients, A and B are negatively correlated
hospital_patients = pd.DataFrame({
    'disease_a': disease_a[in_hospital],
    'disease_b': disease_b[in_hospital]
})

print("Correlation in general population:",
      np.corrcoef(disease_a, disease_b)[0, 1])
print("Correlation in hospital:",
      hospital_patients['disease_a'].corr(hospital_patients['disease_b']))
```

### Survivor Bias

Survivor bias is a form of selection bias where we only observe units that "survived" a selection process.

**Example**: During World War II, the military examined returning bombers and found more bullet holes in the wings and fuselage than in the engines. They concluded that the wings and fuselage needed more armor.

Abraham Wald, a statistician, pointed out the flaw: they were only looking at planes that *survived*. The planes hit in the engines didn't return. The absence of bullet holes in the engines among survivors was evidence that engine hits were the most dangerous.

---

## 1.9 Counterfactual Thinking

### The Heart of Causal Inference

The fundamental concept of causal inference is the **counterfactual**: what would have happened if the treatment had been different?

Consider a patient who took a drug and recovered. The causal question is: **would the patient have recovered if they hadn't taken the drug?**

This question has no direct answer—we can't observe both outcomes for the same patient at the same time. This is the **Fundamental Problem of Causal Inference**.

### The Fundamental Problem

For any individual unit $i$:
- Let $Y_i(1)$ be the outcome if unit $i$ receives treatment
- Let $Y_i(0)$ be the outcome if unit $i$ does not receive treatment
- The individual treatment effect is: $\tau_i = Y_i(1) - Y_i(0)$

We can only observe one of these potential outcomes. If unit $i$ receives treatment, we observe $Y_i(1)$ but not $Y_i(0)$. The unobserved outcome is the counterfactual.

### Rubin Causal Model

Donald Rubin formalized this framework, now called the **Rubin Causal Model** or **Potential Outcomes Framework**.

The key insight: causal inference is fundamentally about **missing data**. The counterfactual outcome is missing, and we need methods to fill in what we would have observed under the alternative treatment.

### How We Solve the Problem

Since we can't observe individual causal effects, we work with **population-level quantities**:

1. **Average Treatment Effect (ATE)**: $E[Y(1) - Y(0)]$
2. **Average Treatment Effect on the Treated (ATT)**: $E[Y(1) - Y(0) | T = 1]$
3. **Average Treatment Effect on the Untreated (ATC)**: $E[Y(1) - Y(0) | T = 0]$

We estimate these quantities using:
- **Randomized experiments**: Random assignment ensures that treatment and potential outcomes are independent
- **Observational studies**: We must make assumptions (like exchangeability) to adjust for confounding

---

## 1.10 Pearl's Ladder of Causation

### Three Levels of Reasoning

Judea Pearl's Ladder of Causation provides a framework for understanding what types of questions can be answered with different types of data and methods.

#### Level 1: Association (Seeing)

**Question**: What is the probability of Y given that I observed X?

**Notation**: $P(Y | X)$

**Methods**: Statistics, machine learning, predictive modeling

**Example**: "What is the probability of lung cancer given that someone smokes?"

**Limitation**: Association tells us nothing about what will happen if we *change* X.

#### Level 2: Intervention (Doing)

**Question**: What happens to Y if I do X?

**Notation**: $P(Y | do(X))$

**Methods**: Randomized experiments, causal inference methods

**Example**: "What happens to lung cancer rates if we ban smoking?"

**Key distinction**: $P(Y | X) \neq P(Y | do(X))$ in general. Observing that smokers have cancer doesn't mean forcing people to smoke would cause cancer (though in this case it would, because smoking does cause cancer).

#### Level 3: Counterfactuals (Imagining)

**Question**: What would have happened to Y if X had been different?

**Notation**: $P(Y_x | X = x', Y = y')$

**Methods**: Structural causal models, counterfactual reasoning

**Example**: "Would this patient have survived if they had received the drug?"

**Application**: Used in attribution, blame, credit, and understanding specific cases.

### The Insufficiency of Data

The critical insight from Pearl's framework: **data alone is not enough for causal inference.**

- Level 1 questions can be answered from data alone
- Level 2 questions require additional causal assumptions (encoded in a DAG)
- Level 3 questions require even stronger assumptions

This is why causal inference requires more than just data—it requires domain knowledge, assumptions, and careful reasoning.

---

## 1.11 Case Study: The Framingham Heart Study

### Background

The Framingham Heart Study, started in 1948, is one of the longest-running and most influential epidemiological studies in history. It has followed multiple generations of residents in Framingham, Massachusetts, to identify risk factors for cardiovascular disease.

### What They Found

The study identified smoking, high cholesterol, and high blood pressure as major risk factors for heart disease. These findings transformed public health policy and medical practice.

### The Causal Question

But here's the subtle point: **the Framingham Heart Study is observational, not experimental.** Nobody randomly assigned people to smoke or not smoke.

So how did they establish that smoking *causes* heart disease?

The answer involves:
1. **Temporal ordering**: Smoking preceded heart disease
2. **Dose-response**: More smoking → more heart disease
3. **Biological mechanism**: We understand how smoking damages blood vessels
4. **Consistency**: Multiple studies in different populations found the same result
5. **Control of confounders**: They measured and adjusted for potential confounders

This is an example of the **Bradford Hill criteria** for causation—guidelines for evaluating whether an association is likely causal when experiments aren't possible.

### The Modern Perspective

Today, we would approach this question using more sophisticated causal inference methods:
- Propensity score matching to create comparable groups
- Instrumental variables to address unmeasured confounding
- Sensitivity analysis to assess how strong unmeasured confounding would need to be to explain away the results

But the fundamental logic is the same: we're trying to answer a causal question with observational data.

---

## 1.12 Python Workshop: Ice Cream and Drowning Simulation

Let's work through a complete example to understand the difference between correlation and causation.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
n = 1000

# Temperature is the common cause
temperature = np.random.normal(25, 8, n)
temperature = np.clip(temperature, 0, 45)  # Realistic temperature range

# Ice cream sales increase with temperature
ice_cream_sales = 50 + 2 * temperature + np.random.normal(0, 10, n)
ice_cream_sales = np.maximum(ice_cream_sales, 0)

# Drowning deaths increase with temperature (more swimming)
drowning_deaths = 0.5 + 0.1 * temperature + np.random.normal(0, 2, n)
drowning_deaths = np.maximum(drowning_deaths, 0)

# Create dataframe
df = pd.DataFrame({
    'temperature': temperature,
    'ice_cream_sales': ice_cream_sales.round(0),
    'drowning_deaths': drowning_deaths.round(0).astype(int)
})

print("Sample data:")
print(df.head(10))
print("\nBasic statistics:")
print(df.describe())
```

### Examining the Correlation

```python
# Correlation between ice cream and drowning
corr = df['ice_cream_sales'].corr(df['drowning_deaths'])
print(f"Correlation between ice cream sales and drowning deaths: {corr:.3f}")

# But is this causal?
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Ice cream vs drowning (spurious correlation)
axes[0].scatter(df['ice_cream_sales'], df['drowning_deaths'],
                alpha=0.5, s=10, c='red')
axes[0].set_xlabel('Ice Cream Sales')
axes[0].set_ylabel('Drowning Deaths')
axes[0].set_title(f'Ice Cream vs Drowning\nr = {corr:.3f}')

# Plot 2: Temperature vs ice cream (true relationship)
axes[1].scatter(df['temperature'], df['ice_cream_sales'],
                alpha=0.5, s=10, c='orange')
axes[1].set_xlabel('Temperature')
axes[1].set_ylabel('Ice Cream Sales')
axes[1].set_title('Temperature causes Ice Cream Sales')

# Plot 3: Temperature vs drowning (true relationship)
axes[2].scatter(df['temperature'], df['drowning_deaths'],
                alpha=0.5, s=10, c='blue')
axes[2].set_xlabel('Temperature')
axes[2].set_ylabel('Drowning Deaths')
axes[2].set_title('Temperature causes Drowning')

plt.tight_layout()
plt.savefig('../figures/01-correlation-causation.png', dpi=150)
plt.show()
```

### Controlling for the Confounder

```python
# When we control for temperature, the association disappears
from sklearn.linear_model import LinearRegression

# Simple regression (confounded)
X_simple = df[['ice_cream_sales']]
y = df['drowning_deaths']
reg_simple = LinearRegression().fit(X_simple, y)
print(f"Simple regression coefficient: {reg_simple.coef_[0]:.3f}")

# Multiple regression (controlling for temperature)
X_multiple = df[['ice_cream_sales', 'temperature']]
reg_multiple = LinearRegression().fit(X_multiple, y)
print(f"Ice cream coefficient after controlling for temperature: {reg_multiple.coef_[0]:.3f}")
print(f"Temperature coefficient: {reg_multiple.coef_[1]:.3f}")

# The ice cream coefficient should be close to zero
# because temperature is the confounder
```

### Key Takeaway

The correlation between ice cream sales and drowning deaths is real, but it's not causal. Temperature is a confounder that causes both. When we control for temperature, the apparent association between ice cream and drowning disappears.

This is why we need causal inference methods—correlation alone can be deeply misleading.

---

## 1.13 Common Mistakes

### Mistake 1: Confusing Correlation with Causation

**The error**: Assuming that because two things are correlated, one causes the other.

**Why it happens**: Our brains are wired to find patterns and infer causation. This was useful for survival but misleading for data analysis.

**How to avoid it**: Always ask: "What are the alternative explanations for this correlation?" Consider confounding, reverse causation, selection bias, and coincidence.

### Mistake 2: Ignoring Confounders

**The error**: Failing to account for variables that affect both treatment and outcome.

**Why it happens**: It's easy to focus on the variables of interest and forget about the broader context.

**How to avoid it**: Draw a DAG. Identify potential confounders. Control for them in your analysis.

### Mistake 3: Overclaiming from Observational Data

**The error**: Making strong causal claims based on observational data without acknowledging the assumptions required.

**Why it happens**: There's pressure to have definitive answers, and hedging feels weak.

**How to avoid it**: Be explicit about your assumptions. Conduct sensitivity analysis. Use language like "consistent with" rather than "proves."

### Mistake 4: Confusing Statistical Significance with Practical Importance

**The error**: Assuming that a statistically significant result is practically important.

**Why it happens**: Statistical significance is easier to compute than practical significance.

**How to avoid it**: Report effect sizes and confidence intervals. Ask: "Is this effect large enough to matter in practice?"

---

## 1.14 Discussion Questions

1. **The Chocolate Paradox**: Countries that consume more chocolate produce more Nobel Prize winners. Is this causal? What are the alternative explanations?

2. **Medical Decision-Making**: A doctor observes that patients who take a certain supplement have better health outcomes. Should the doctor recommend this supplement? What additional information would you want?

3. **Policy Evaluation**: A city implements a new traffic safety program and accidents decrease. The mayor claims the program "caused" the reduction. What would you need to know to evaluate this claim?

4. **Personal Decisions**: You notice that you perform better on exams after getting a good night's sleep. Should you always prioritize sleep before exams? What other factors might be at play?

5. **The Ethics of Causal Claims**: When is it responsible to make a causal claim based on observational data? When is it irresponsible?

---

## 1.15 Interview Questions

### Question 1
**"What is the difference between correlation and causation? Give an example."**

**Model Answer**: Correlation is a statistical association between two variables—when one changes, the other tends to change. Causation means one variable actually influences the other. Ice cream sales and drowning deaths are correlated because both increase in summer, but ice cream doesn't cause drowning—heat is a confounder that causes both. The key distinction is that correlation measures association while causation implies that manipulating one variable would change the other.

### Question 2
**"Why can't we just use machine learning to determine causation?"**

**Model Answer**: Machine learning excels at prediction—it finds patterns in data that help forecast outcomes. But prediction is Level 1 on Pearl's Ladder of Causation. Causal inference is Level 2—it asks what would happen if we intervened. A model that predicts heart attacks based on cholesterol doesn't tell us whether reducing cholesterol would reduce heart attacks. The model might rely on variables that correlate with heart attacks but aren't causes. Causal inference requires additional assumptions about the data-generating process that standard ML doesn't incorporate.

### Question 3
**"What is confounding? How would you explain it to a non-technical audience?"**

**Model Answer**: Confounding is when a third variable affects both what we're studying and the outcome, making it look like there's a relationship when there isn't, or making a relationship look stronger or weaker than it really is. Imagine you notice that people who carry lighters have higher lung cancer rates. But lighters don't cause cancer—smoking does, and smokers carry lighters. Smoking is the confounder. To understand the true effect of lighters on cancer, you'd need to compare smokers who carry lighters to smokers who don't, or control for smoking status.

### Question 4
**"What is a counterfactual? Why is it important?"**

**Model Answer**: A counterfactual is what would have happened if something had been different. For a patient who took a drug and recovered, the counterfactual is: would they have recovered without the drug? This is the heart of causal inference because the causal effect is the difference between what happened and what would have happened under the alternative treatment. The fundamental problem is that we can never observe both outcomes for the same individual, so causal inference is inherently about reasoning about unobserved possibilities.

### Question 5
**"Design a study to determine whether a new education program improves student test scores. What are the key challenges?"**

**Model Answer**: The gold standard would be a randomized controlled trial where students are randomly assigned to the program or control group. Key challenges include: (1) getting ethical approval to withhold potentially beneficial education, (2) ensuring compliance—students assigned to the program actually participate, (3) preventing contamination—control students getting similar benefits elsewhere, (4) measuring outcomes appropriately—test scores may not capture all program benefits, (5) maintaining long-term follow-up, and (6) external validity—whether results from one setting generalize to others.

---

## 1.16 Knowledge Check

### Multiple Choice

1. **Which of the following best describes a confounder?**
   - A) A variable caused by the treatment
   - B) A variable caused by the outcome
   - C) A common cause of both treatment and outcome
   - D) A variable unrelated to either treatment or outcome

2. **What is the fundamental problem of causal inference?**
   - A) We can't measure variables accurately
   - B) We can never observe both potential outcomes for the same unit
   - C) Statistical tests are underpowered
   - D) Samples are too small

3. **In Pearl's Ladder of Causation, what level does "What happens to Y if I do X?" correspond to?**
   - A) Level 1: Association
   - B) Level 2: Intervention
   - C) Level 3: Counterfactuals
   - D) None of the above

4. **Reverse causation occurs when:**
   - A) The cause and effect are the same variable
   - B) We mistakenly assume X causes Y when actually Y causes X
   - C) There is no causal relationship
   - D) The causal effect is zero

5. **Which of the following is NOT a reason correlation doesn't imply causation?**
   - A) Confounding
   - B) Reverse causation
   - C) Selection bias
   - D) Large sample size

### Short Answer

6. **Explain why randomized experiments are considered the gold standard for causal inference.**

7. **Describe a real-world example of Berkson's paradox.**

8. **Why can't machine learning alone answer causal questions?**

9. **What is the difference between ATE and ATT?**

10. **Draw a DAG showing the relationship between smoking, exercise, and heart disease.**

### Critical Thinking

11. **A study finds that people who eat breakfast have lower BMI. Is this causal? What alternative explanations exist? Design an analysis to test the causal claim.**

12. **Explain why the Bradford Hill criteria are useful but insufficient for establishing causation from observational data.**

---

## 1.17 Practical Checklist

Before making any causal claim, check:

- [ ] Have I clearly defined the treatment and outcome?
- [ ] Have I identified potential confounders?
- [ ] Have I drawn a DAG to represent my causal assumptions?
- [ ] Is my causal assumption plausible given existing knowledge?
- [ ] Have I controlled for confounders in my analysis?
- [ ] Have I checked for reverse causation?
- [ ] Have I considered selection bias?
- [ ] Have I conducted sensitivity analysis?
- [ ] Have I been honest about what my analysis can and cannot conclude?
- [ ] Have I distinguished between statistical significance and practical importance?

---

## Watch and Connect

```{admonition} Recommended lecture
:class: tip

[**Course motivation: why causal inference?**](https://www.youtube.com/watch?v=CfzO4IEMVUk&list=PLoazKTcS0Rzb6bb9L508cyJ1z-U9iWkA0&index=1)

Brady Neal introduces the central distinction between association and intervention.

**Active-viewing prompt:** Write down one prediction question and one causal question that use the same variables.
```

---

## 1.18 Summary

In this lesson, we established the foundations of causal inference:

1. **Causation is different from correlation.** Two variables can be correlated without one causing the other.

2. **Three main threats to causal inference**: confounding, reverse causation, and selection bias.

3. **Counterfactual reasoning** is the heart of causal inference: what would have happened under a different treatment?

4. **The Fundamental Problem**: We can never observe both potential outcomes for the same unit, so causal inference requires assumptions.

5. **Pearl's Ladder of Causation**: Association, intervention, and counterfactuals are three distinct levels that require different methods and assumptions.

6. **Decisions require causal knowledge**, not just predictive models.

The tools and concepts introduced here will be formalized in the following lessons. In Lesson 2, we'll dive deeper into the potential outcomes framework.

---

## 1.19 Further Reading

### Classic Works
- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*. Cambridge University Press.
- Rubin, D.B. (2005). "Causal Inference Using Potential Outcomes." *Journal of the American Statistical Association*.
- Holland, P.W. (1986). "Statistics and Causal Inference." *Journal of the American Statistical Association*.

### Modern Textbooks
- Cunningham, S. (2021). *Causal Inference: The Mixtape*. MIT Press.
- Hernán, M.A. & Robins, J.M. (2020). *Causal Inference: What If*. Chapman and Hall/CRC.
- Huntington-Klein, N. (2021). *The Effect*. Chapman and Hall/CRC.

### Accessible Introductions
- Pearl, J. & Mackenzie, D. (2018). *The Book of Why*. Basic Books.
- Taleb, N.N. (2007). *The Black Swan*. Random House.

### Online Resources
- [Causal Inference: The Mixtape](https://mixtape.scunning.com/)
- [Brady Neal's Introduction to Causal Inference](https://www.bradyneal.com/causal-inference-course)
- [Judea Pearl's Causal Inference Resources](https://causality.cs.ucla.edu/)


---

## Worked Examples

### Example 1: The Ice Cream Drowning Fallacy

Ice cream sales correlate with drowning deaths. Does ice cream cause drowning? No—temperature is a confounder. Both ice cream consumption and swimming (which causes drowning risk) increase in summer.

```python
import numpy as np
import pandas as pd

np.random.seed(42)
n = 500
temperature = np.random.uniform(60, 100, n)
ice_cream_sales = 0.8 * temperature + np.random.normal(0, 5, n)
drowning_deaths = 0.05 * temperature + np.random.normal(0, 2, n)

df = pd.DataFrame({'temperature': temperature, 'ice_cream': ice_cream_sales, 'drowning': drowning_deaths})
print(f"Correlation (ice cream, drowning): {df['ice_cream'].corr(df['drowning']):.3f}")
print("But controlling for temperature eliminates the association!")
```

### Example 2: Simpson's Paradox in Action

A treatment appears harmful overall but beneficial in every subgroup. This happens when subgroups have different baseline risks.

```python
import pandas as pd

# Create Simpson's Paradox data
data = {
    'group': ['A']*400 + ['B']*600,
    'severity': ['mild']*100 + ['severe']*300 + ['mild']*500 + ['severe']*100,
    'recovered': [80, 90, 400, 50]
}
df = pd.DataFrame(data)
print("Overall: Treatment A has higher recovery rate")
print("But within each severity group, Treatment B is better!")
```


---

## Exercises

### Exercise 1

Identify the confounders in a study examining the relationship between coffee consumption and heart disease. What variables might explain the observed association?

### Exercise 2

Using the ice cream/drowning example, write Python code to demonstrate how controlling for temperature changes the estimated association.

### Exercise 3

Find a real-world example of Simpson's Paradox and explain the causal structure that produces it.
