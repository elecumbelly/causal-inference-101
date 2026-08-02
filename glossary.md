# Glossary

## A

**ATE (Average Treatment Effect)**
The average difference in potential outcomes across the entire population. Formally, $ATE = E[Y(1) - Y(0)]$ where $Y(1)$ is the outcome under treatment and $Y(0)$ is the outcome under control.

**ATC (Average Treatment Effect on the Treated)**
The average treatment effect among those who actually received treatment. Formally, $ATC = E[Y(1) - Y(0) | T = 1]$.

**ATT (Average Treatment Effect on the Untreated)**
The average treatment effect among those who did not receive treatment. Formally, $ATT = E[Y(1) - Y(0) | T = 0]$.

**Attrition**
The loss of participants from a study over time. Attrition can bias results if the reason for dropping out is related to the treatment or outcome.

**Instrument**
A variable that affects the outcome only through its effect on the treatment. Instruments must satisfy relevance and exclusion restriction assumptions.

---

## B

**Backdoor Criterion**
A graphical criterion for identifying causal effects. A set of variables $Z$ satisfies the backdoor criterion relative to $X$ and $Y$ if: (1) no node in $Z$ is a descendant of $X$, and (2) $Z$ blocks every path between $X$ and $Y$ that contains an arrow into $X$.

**Backdoor Path**
A path between two variables that contains an arrow pointing into the treatment variable. Backdoor paths represent confounding.

**Balance**
In propensity score methods, the degree to which the distribution of covariates is similar between treated and control groups after weighting or matching.

**Bandwidth**
In regression discontinuity, the window around the cutoff used for estimation. The choice of bandwidth involves a bias-variance tradeoff.

**Berkson's Paradox**
A form of selection bias where conditioning on a common effect of two independent causes creates a spurious association between them.

---

## C

**Causal Effect**
The difference between what happens when a unit receives a treatment and what would have happened if the same unit had not received the treatment, holding all else constant.

**Collider**
A variable that is jointly caused by two other variables (i.e., it has two incoming arrows in a DAG). Conditioning on a collider can create spurious associations.

**Collider Bias**
The distortion in estimated associations that arises from conditioning on a collider variable.

**Conditional Independence**
Two variables $X$ and $Y$ are conditionally independent given $Z$ if knowing $Z$ makes $X$ and $Y$ independent. Written $X \perp\!\!\!\perp Y | Z$.

**Consistency**
The assumption that the observed outcome equals the potential outcome corresponding to the observed treatment. i.e., if $T_i = t$, then $Y_i = Y_i(t)$.

**Confidence Interval**
A range of values that is likely to contain the true parameter with a specified probability (typically 95%).

**Confounding**
The presence of a common cause of both the treatment and outcome, creating a non-causal association between them.

**Confounding Variable**
A variable that causes both the treatment and outcome, creating a spurious association.

**Counterfactual**
A hypothetical scenario describing what would have happened under a different treatment assignment. The cornerstone of causal reasoning.

**Covariate**
An observed variable that may affect the treatment assignment, the outcome, or both.

---

## D

**DAG (Directed Acyclic Graph)**
A graphical representation of causal relationships where nodes represent variables and directed edges represent direct causal effects. "Acyclic" means there are no feedback loops.

**D-Separation**
A graphical criterion for determining conditional independence in a DAG. Two sets of variables are d-separated if every path between them is "blocked" by observed variables.

**Difference-in-Differences (DiD)**
A method for estimating causal effects by comparing the change in outcomes over time between a treatment group and a control group.

**Doubly Robust Estimation**
An estimation method that provides consistent estimates if either the outcome model or the treatment model (propensity score) is correctly specified, but not necessarily both.

---

## E

**Effect Modification**
When the treatment effect varies across levels of another variable. Also called heterogeneous treatment effects.

**Endogeneity**
A situation where the treatment variable is correlated with the error term in a regression model, typically due to omitted variables, simultaneity, or measurement error.

**Exchangeability**
The assumption that the treatment assignment is independent of potential outcomes, conditional on observed covariates. Also called unconfoundedness.

**Exclusion Restriction**
In instrumental variables, the assumption that the instrument affects the outcome only through its effect on the treatment.

---

## F

**Falsifiability**
The principle that a hypothesis must be capable of being proven false by evidence.

**Fixed Effects**
A modeling approach that controls for time-invariant unobserved heterogeneity by including individual-specific intercepts.

**Fork**
A DAG structure where a common cause points to two effects: $X \leftarrow Z \rightarrow Y$. Conditioning on $Z$ blocks the association between $X$ and $Y$.

**Frontdoor Criterion**
A graphical criterion for identifying causal effects when backdoor adjustment is not possible. Requires a mediator that intercepts all causal paths from treatment to outcome.

---

## G

**G-computation**
A method for estimating causal effects of time-varying treatments by modeling the joint distribution of potential outcomes.

---

## H

**Heterogeneity**
Variation in treatment effects across individuals or subgroups.

**Heterogeneous Treatment Effects**
Treatment effects that differ across individuals based on their characteristics.

---

## I

**Imputation**
The process of filling in missing values in a dataset.

**Instrumental Variable (IV)**
A variable that affects the outcome only through its effect on the treatment. Used to estimate causal effects when treatment assignment is endogenous.

**Inverse Probability Weighting (IPW)**
A method that creates a pseudo-population where treatment assignment is independent of observed covariates by weighting each observation by the inverse of its probability of receiving its actual treatment.

**Intention-to-Treat (ITT)**
An analysis that compares outcomes based on the treatment assignment, regardless of whether participants actually received the treatment.

**Interference**
When one unit's treatment affects another unit's outcome, violating the SUTVA assumption.

---

## L

**LATE (Local Average Treatment Effect)**
The average treatment effect among "compliers"—units whose treatment status is affected by the instrument. Also called the complier average causal effect (CACE).

**Lurking Variable**
An unobserved variable that may be confounding the relationship between treatment and outcome.

---

## M

**M-Bias**
A specific DAG structure where conditioning on a mediator can introduce bias rather than remove it.

**Marginal Structural Models (MSMs)**
Models that describe the relationship between time-varying treatments and potential outcomes in the marginal (population-averaged) distribution.

**Mediator**
A variable that lies on the causal pathway between treatment and outcome. The treatment affects the mediator, which in turn affects the outcome.

**Mediation Analysis**
The decomposition of a total causal effect into direct and indirect effects through mediators.

**Missing at Random (MAR)**
Missingness depends only on observed data, not on the missing values themselves.

**Missing Completely at Random (MCAR)**
Missingness is independent of all observed and unobserved data.

**Missing Not at Random (MNAR)**
Missingness depends on the missing values themselves.

---

## N

**Natural Experiment**
An empirical setting where treatment assignment is determined by factors outside the researcher's control, resembling random assignment.

**Negative Control**
An outcome or treatment that is known to be unaffected by the treatment, used to test for violations of study assumptions.

---

## O

**Omitted Variable Bias**
Bias in estimated treatment effects caused by failing to control for a variable that affects both treatment and outcome.

**Overlap**
The assumption that every unit has a positive probability of receiving either treatment. Also called positivity.

---

## P

**Path Analysis**
The study of causal pathways in a DAG, including direct and indirect effects.

**Placebo Test**
A test that checks whether the estimated effect appears in settings where no effect should exist. Used to validate identification strategies.

**Positivity**
The assumption that for every combination of covariate values, there is a positive probability of receiving each treatment level.

**Potential Outcomes**
The outcomes that would be observed under different treatment levels. The foundation of the Rubin Causal Model.

**Propensity Score**
The probability of receiving treatment conditional on observed covariates: $e(X) = P(T = 1 | X)$. Used in matching, stratification, and weighting.

---

## R

**Randomized Controlled Trial (RCT)**
A study where units are randomly assigned to treatment or control groups, ensuring that potential outcomes are independent of treatment assignment.

**Regression Discontinuity (RD)**
A method that exploits a cutoff rule that assigns treatment based on whether a running variable exceeds a threshold.

**Regression Adjustment**
Using regression to control for confounders and estimate causal effects.

---

## S

**Selection Bias**
Systematic differences between those who are selected into a study (or a treatment) and those who are not.

**Sensitivity Analysis**
An analysis that examines how conclusions change under different assumptions about unobserved confounding.

**Simpson's Paradox**
A phenomenon where an association between two variables reverses when a third variable is taken into account.

**SUTVA (Stability Unit Treatment Value Assumption)**
The assumption that one unit's treatment does not affect another unit's outcome, and that there is only one version of each treatment level.

**Synthetic Control**
A method for estimating causal effects in comparative case studies by constructing a weighted combination of control units that approximates the treated unit's pre-treatment trajectory.

---

## T

**Treatment**
The intervention or exposure whose causal effect on the outcome is of interest.

**Treatment Effect**
The difference in potential outcomes: $Y(1) - Y(0)$.

---

## U

**Unconfoundedness**
The assumption that, conditional on observed covariates, treatment assignment is independent of potential outcomes. Also called exchangeability.

**Unobserved Confounding**
Confounding caused by variables that are not measured in the dataset.

---

## V

**Valid Instrument**
An instrument that satisfies both the relevance condition (correlated with treatment) and the exclusion restriction (affects outcome only through treatment).

---

## W

**Weighting**
In causal inference, methods that reweight observations to create a pseudo-population where treatment assignment is independent of confounders.
