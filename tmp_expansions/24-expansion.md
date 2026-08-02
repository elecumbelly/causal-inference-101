

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
