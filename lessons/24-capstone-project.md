---
title: "Lesson 24: Capstone Project"
subtitle: "From First Principles to Modern Causal Inference"
---

# Lesson 24: Capstone Project
## Opening Story: The Capstone Challenge

You have spent the entire course building a toolkit of causal inference methods. Now it is time to put everything together. This capstone lesson presents a real-world policy evaluation challenge that requires you to:

1. Identify the causal question
2. Draw causal diagrams
3. Choose appropriate identification strategy
4. Implement the analysis
5. Assess robustness
6. Communicate findings

The challenge: Evaluate the effect of a job training program on earnings, using observational data with multiple potential confounders.

---

## Learning Objectives

By the end of this lesson, you should be able to:

1. Synthesize multiple causal inference methods
2. Choose appropriate methods for complex evaluation questions
3. Handle multiple confounders and potential violations
4. Communicate causal findings to diverse audiences
5. Design a complete causal analysis from start to finish


```{figure} ../figures/instructional/causal-workflow.svg
:name: lesson-24-causal-workflow
:alt: A defensible analysis moves from question to interpretation without skipping identification.
:width: 100%

A defensible analysis moves from question to interpretation without skipping identification.
```

---

## Overview

This capstone project synthesizes everything you've learned throughout the course. You will conduct a complete causal analysis of a real-world policy question, from formulating the problem to interpreting results and policy recommendations.

---

## Project Options

Choose ONE of the following projects:

### Option 1: Effect of Minimum Wage on Employment

Using the Card & Krueger (1994) dataset or similar data, estimate the effect of minimum wage increases on employment in the fast-food industry.

**Methods to consider:**
- Difference-in-differences
- Synthetic control
- Regression discontinuity (if applicable)

### Option 2: Effect of Education on Earnings

Using the National Longitudinal Survey of Youth (NLSY) or similar data, estimate the causal return to education.

**Methods to consider:**
- Instrumental variables
- Propensity score matching
- Sensitivity analysis

### Option 3: Effect of Healthcare Policy on Health Outcomes

Using data from the Oregon Health Insurance Experiment or similar, estimate the effect of health insurance on health outcomes.

**Methods to consider:**
- Randomized controlled trial analysis
- Regression discontinuity
- Heterogeneous treatment effects

### Option 4: Effect of Environmental Policy on Pollution

Using data from California's cap-and-trade program or similar, estimate the effect of environmental regulation on pollution.

**Methods to consider:**
- Difference-in-differences
- Synthetic control
- Event study

---

## Project Requirements

### Part 1: Problem Formulation (10 points)

1. **Research question**: Clearly state the causal question
2. **Causal model**: Draw a DAG of the causal relationships
3. **Identification strategy**: Explain how you will identify the causal effect
4. **Data description**: Describe the data source and variables

### Part 2: Data Analysis (20 points)

1. **Descriptive statistics**: Summarize the data
2. **Visualizations**: Create informative plots
3. **Balance checks**: Assess covariate balance (if applicable)
4. **Main results**: Estimate the causal effect using appropriate methods
5. **Robustness checks**: Conduct sensitivity analysis

### Part 3: Interpretation and Policy (15 points)

1. **Interpret results**: Explain what the estimates mean
2. **Policy implications**: Discuss what the results suggest for policy
3. **Limitations**: Acknowledge threats to validity
4. **Future research**: Suggest extensions

### Part 4: Code and Reproducibility (5 points)

1. **Clean code**: Well-organized and commented
2. **Reproducible**: Someone else can run your analysis
3. **Documentation**: Clear README explaining how to run

---

## Example Project: Minimum Wage and Employment

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Simulate Card & Krueger style data
np.random.seed(42)

# New Jersey (treated) and Pennsylvania (control)
n_nj = 100
n_pa = 100

# Pre-treatment employment
pre_nj = np.random.normal(20, 3, n_nj)
pre_pa = np.random.normal(21, 3, n_pa)

# Treatment effect (minimum wage increase)
treatment_effect = -0.5  # Small negative effect

# Post-treatment employment
post_nj = pre_nj + treatment_effect + np.random.normal(0, 2, n_nj)
post_pa = pre_pa + np.random.normal(0, 2, n_pa)

# Difference-in-differences
did_estimate = (post_nj.mean() - pre_nj.mean()) - (post_pa.mean() - pre_pa.mean())

print("Difference-in-Differences Estimate:")
print(f"  New Jersey change: {post_nj.mean() - pre_nj.mean():.2f}")
print(f"  Pennsylvania change: {post_pa.mean() - pre_pa.mean():.2f}")
print(f"  DiD estimate: {did_estimate:.2f}")
print(f"  True effect: {treatment_effect}")

# Event study
fig, ax = plt.subplots(figsize=(10, 6))
# ... plot event study coefficients
plt.title("Effect of Minimum Wage on Employment")
plt.xlabel("Time relative to policy change")
plt.ylabel("Employment")
plt.show()
```

---

## The Six-Stage Causal Workflow

Use this sequence for every project—not just the capstone:

| Stage | Required output | Failure it prevents |
|---|---|---|
| Question | A treatment, outcome, population and time horizon | Vague causal language |
| Estimand | A formal target such as ATE, ATT or LATE | Estimating the wrong effect |
| Identification | A DAG or design argument plus assumptions | Treating an estimator as a design |
| Estimation | A method matched to the estimand and data | Technique-first analysis |
| Diagnostics | Overlap, balance, pre-trends, fit or falsification checks | Hidden design failure |
| Interpretation | Magnitude, uncertainty, limitations and transportability | Overclaiming |

Before presenting a number, complete an **assumption audit**: name each assumption, explain why it might hold, show any available diagnostic, and describe the direction of bias if it fails. Diagnostics can challenge assumptions; they rarely prove them.

## Grading Rubric

| Criterion | Excellent (90-100%) | Good (80-89%) | Satisfactory (70-79%) | Needs Improvement (<70%) |
|-----------|---------------------|---------------|----------------------|--------------------------|
| Problem Formulation | Clear question, valid DAG, strong identification | Good question, reasonable DAG | Basic question, incomplete DAG | Unclear question, no DAG |
| Data Analysis | Appropriate methods, thorough checks | Good methods, basic checks | Limited methods, minimal checks | Inappropriate methods |
| Interpretation | Nuanced, acknowledges limitations | Good interpretation | Basic interpretation | Superficial or incorrect |
| Code Quality | Clean, reproducible, documented | Mostly clean, reproducible | Functional but messy | Difficult to run |

---

## Timeline

- **Week 1**: Choose project, formulate question, obtain data
- **Week 2**: Conduct exploratory data analysis
- **Week 3**: Implement main analysis
- **Week 4**: Complete robustness checks and write-up
- **Week 5**: Final presentation and submission

---

## Resources

### Data Sources

- **Card & Krueger**: Available from David Card's website
- **Oregon Health Insurance Experiment**: Oregon Health Insurance Experiment website
- **California Tobacco Control**: CDC STATE data
- **NLSY**: Bureau of Labor Statistics

### Software

- Python: numpy, pandas, statsmodels, econml, linearmodels
- R: did, synth, rdrobust, ivreg

### References

- Angrist, J.D. & Pischke, J.S. (2009). *Mostly Harmless Econometrics*. Princeton University Press.
- Cunningham, S. (2021). *Causal Inference: The Mixtape*. MIT Press.
- Huntington-Klein, N. (2022). *The Effect*. CRC Press.

---

## Submission

Submit the following:

1. **Jupyter notebook** with complete analysis
2. **README.md** explaining the project
3. **requirements.txt** with all dependencies
4. **10-minute presentation** (video or slides)

---

## Final Thoughts

Causal inference is both an art and a science. The methods you've learned provide powerful tools for answering causal questions, but they require judgment, domain knowledge, and critical thinking. As you apply these methods in your own work, remember:

1. **No method is perfect**: Every study has limitations
2. **Assumptions matter**: Always discuss what assumptions are needed
3. **Transparency is key**: Document your decisions and share your code
4. **Context matters**: The same method can give different results in different settings

Good luck with your capstone project!

---

## Watch and Connect

```{admonition} Recommended lecture
:class: tip

[**Course overview: the causal workflow**](https://www.youtube.com/watch?v=CfzO4IEMVUk&list=PLoazKTcS0Rzb6bb9L508cyJ1z-U9iWkA0&index=1)

Return to the opening motivation and use it as a checklist for the capstone.

**Active-viewing prompt:** Can you state the question, estimand, identification assumptions, estimator, and limits in five sentences?
```

---

---

## Course Summary

Throughout this course, you've learned:

1. **Foundations**: Potential outcomes, DAGs, confounding
2. **Methods**: RCTs, regression, matching, IV, DiD, RDD, synthetic control
3. **Advanced topics**: Mediation, heterogeneous effects, sensitivity analysis, longitudinal data
4. **Modern approaches**: Causal discovery, machine learning, Bayesian methods
5. **Applications**: External validity, fairness, decision theory

You now have the tools to answer causal questions rigorously. The challenge is to use them wisely.

---

## Further Reading

- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*. Cambridge University Press.
- Rubin, D.B. (2005). "Causal Inference Using Potential Outcomes." *Journal of the American Statistical Association*.
- Hernán, M.A. & Robins, J.M. (2020). *Causal Inference: What If*. Chapman & Hall/CRC.


---

## Worked Examples

### Example 1: Comprehensive Policy Evaluation

A government wants to evaluate a new education policy. The analysis combines:
- RCT evidence from a pilot study
- Observational data for long-term effects
- Causal inference methods for mechanism understanding
- Decision theory for policy recommendation

### Example 2: Technology Platform Causal Analysis

A tech company wants to understand user behavior. The analysis combines:
- A/B testing for feature effects
- Observational methods for user retention
- Machine learning for heterogeneous effects
- Fairness analysis for algorithmic impact

### Example 3: Healthcare Treatment Comparison

A hospital system wants to compare treatment protocols. The analysis combines:
- Meta-analysis of existing trials
- Observational data from electronic health records
- Sensitivity analysis for unmeasured confounding
- Decision theory for treatment guidelines

### Example 4: Social Science Research Program

A research team studies social inequality. The analysis combines:
- Natural experiments for causal identification
- Longitudinal methods for dynamic effects
- Causal discovery for mechanism exploration
- External validity assessment for generalizability

---

## Diagnostics: Integration Checklist

### Method Selection

| Question | Method | When to Use |
|----------|--------|-------------|
| What is the causal effect? | RCT, DiD, RDD, IV, PS | Depends on data availability and identification strategy |
| Is the effect heterogeneous? | CATE, meta-learners | When treatment effects vary across subgroups |
| Is the finding robust? | Sensitivity analysis | Always — to assess vulnerability to assumption violations |
| Can it generalize? | Transportability, external validity | When applying findings to new populations |
| Is it fair? | Fairness metrics | When decisions affect different groups differently |

### Quality Assurance

1. **Identification**: Is the causal effect identified? Draw the DAG.
2. **Estimation**: Is the estimator consistent and efficient? Check finite-sample properties.
3. **Inference**: Are confidence intervals valid? Use appropriate standard errors.
4. **Sensitivity**: How robust are findings to assumption violations? Conduct sensitivity analysis.
5. **External validity**: Can findings be generalized? Assess transportability.

---

## Interpretation Workshop

### Integrating Multiple Evidence Sources

- **Triangulation**: If different methods with different assumptions converge, confidence increases
- **Complementarity**: Different methods answer different aspects of the same question
- **Contradiction**: When methods disagree, investigate why — often reveals important insights

### Common Integration Pitfalls

- Cherry-picking the most favorable estimate
- Ignoring assumptions that differ across methods
- Failing to reconcile contradictory findings
- Over-complicating the analysis when simpler methods suffice

---

## Practical Application

### Analysis Protocol

1. **Define the question**: Clearly specify the causal estimand
2. **Map the causal structure**: Draw the DAG
3. **Assess available methods**: What identification strategies are possible?
4. **Select primary method**: Choose the most credible approach
5. **Conduct robustness checks**: Apply alternative methods
6. **Assess external validity**: Can findings generalize?
7. **Present transparently**: Report all findings, not just favorable ones

### Writing a Causal Inference Paper

```python
def paper_structure():
    """Outline for a causal inference paper."""

    sections = {
        'Introduction': [
            'Research question',
            'Why causal inference matters for this question',
            'Brief overview of approach',
        ],
        'Background': [
            'Prior literature',
            'Causal DAG for the setting',
            'Identification strategy',
        ],
        'Data': [
            'Source and collection',
            'Sample construction',
            'Descriptive statistics',
        ],
        'Methods': [
            'Primary estimation strategy',
            'Assumptions and their plausibility',
            'Sensitivity analysis plan',
        ],
        'Results': [
            'Main estimates',
            'Robustness checks',
            'Heterogeneity analysis',
        ],
        'Discussion': [
            'Interpretation of findings',
            'Limitations and threats to validity',
            'External validity assessment',
            'Policy implications',
        ],
    }

    for section, items in sections.items():
        print(f"\n{section}:")
        for item in items:
            print(f"  - {item}")

    return sections
```

---

## Limitations

- No method is perfect: Every approach has assumptions that may fail
- Data limitations: Observational data always leaves room for unmeasured confounding
- Generalizability: Findings may not transport to other settings
- Complexity: Integrating multiple methods requires expertise across disciplines

---

## Exercises

1. **Full analysis**: Conduct a complete causal analysis of a research question using at least three different methods.
2. **Integration**: Write a synthesis comparing findings across methods. Are they consistent?
3. **Sensitivity**: Conduct comprehensive sensitivity analysis. How robust are the findings?
4. **Communication**: Write a policy brief translating causal findings into actionable recommendations.

---

## Projects

### Project 1: Research Replication
Replicate a published causal inference study using the original data and methods. Assess robustness to alternative specifications.

### Project 2: Original Research
Conduct an original causal inference study:
1. Define a research question
2. Identify a causal strategy
3. Collect or access appropriate data
4. Implement the analysis
5. Write up findings following best practices


---


---

## Companion Practice

Move from reading to doing with the complete course materials:

- [Download the worked notebook](../notebooks/24-capstone-project.ipynb)
- [Download the practice lab](../labs/lab24-capstone-project-practice.ipynb)
- [Download the lab solution](../solutions/lab24-solutions.ipynb) — attempt the lab before opening this

The worked notebook demonstrates the lesson's core estimator. The practice lab asks you to make the identification and diagnostic choices yourself.
