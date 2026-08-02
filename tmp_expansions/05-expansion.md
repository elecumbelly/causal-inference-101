

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
