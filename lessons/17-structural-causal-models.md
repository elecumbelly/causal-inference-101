---
title: "Lesson 17: Structural Causal Models"
subtitle: "From First Principles to Modern Causal Inference"
---

# Lesson 17: Structural Causal Models

## Opening Story: Pearl's Revolution

Judea Pearl's work on structural causal models (SCMs) transformed causal inference from a statistical problem to a computational one. By formalizing causal reasoning in mathematical notation, Pearl showed that many causal questions can be answered if we have the right causal model.

SCMs extend DAGs by adding functional equations, allowing us to compute counterfactuals and answer "what if" questions about interventions.

---

## Learning Objectives

By the end of this lesson, you should be able to:

1. Define structural causal models
2. Explain the do-calculus
3. Implement graph-based identification
4. Compute counterfactuals
5. Apply SCMs to complex causal problems

---

## 17.1 The SCM Framework

### Components

An SCM consists of:
1. **Variables**: Endogenous and exogenous
2. **Functions**: Structural equations for each variable
3. **Distribution**: Distribution of exogenous variables

### Example

```
X = f_X(U_X)
Y = f_Y(X, U_Y)
```

The functions are deterministic given the exogenous variables.

---

## 17.2 The Do-Operator

### Definition

$P(Y | do(X = x))$ is the distribution of $Y$ when we intervene to set $X = x$.

### Graph Manipulation

To compute $do(X = x)$:
1. Remove all arrows pointing into $X$
2. Set $X = x$
3. Compute the resulting distribution

```python
import networkx as nx
import numpy as np

# Create a DAG
G = nx.DiGraph()
G.add_edges_from([('X', 'Y'), ('Z', 'X'), ('Z', 'Y')])

def do_operator(G, variable, value):
    """
    Simulate the effect of an intervention.
    """
    # Remove incoming edges
    incoming = list(G.in_edges(variable))
    G.remove_edges_from(incoming)

    # Set variable to value
    # (In a real implementation, this would modify the SCM)

    return G

# Example: Z -> X -> Y, Z -> Y
# do(X) removes Z -> X, keeping only X -> Y
print("Original edges:", list(G.edges()))
G_intervened = do_operator(G.copy(), 'X', 1)
print("After do(X):", list(G_intervened.edges()))
```

---

## 17.3 The Do-Calculus

Three rules for manipulating do-expressions:

1. **Rule 1**: If $Y \perp W | Z, X$ in $G_{\bar{X}}$, then $P(Y | do(X), Z, W) = P(Y | do(X), Z)$

2. **Rule 2**: If $Y \perp W | Z, X$ in $G_{\underline{X}}$, then $P(Y | do(X), Z, W) = P(Y | do(X), Z)$

3. **Rule 3**: If $Y \perp W | Z$ in $G_{\bar{X}\underline{W}}$, then $P(Y | do(X), Z) = P(Y | Z)$

These rules allow us to convert do-expressions to observational distributions when identification is possible.

---

## 17.4 Counterfactuals

### Definition

$Y_{X=x'}$ is the value of $Y$ when we set $X = x'$ in a specific individual's world.

### The Fundamental Law of Counterfactuals

Given an SCM, counterfactuals are computed by:
1. Computing the posterior distribution of exogenous variables given evidence
2. Setting the intervention
3. Computing the downstream effects

---

## 17.5 Common Mistakes

1. **Confusing do and see**: $P(Y | X) \neq P(Y | do(X))$ in general
2. **Ignoring graph structure**: Identification depends on the DAG
3. **Overcomplicating**: Start with simple graphs
4. **Not validating**: Check if the SCM matches domain knowledge

---

## 17.6 Knowledge Check

### Multiple Choice

1. **An SCM consists of:**
   A) Variables and functions
   B) Variables and distributions
   C) Functions and distributions
   D) All of the above

2. **The do-operator:**
   A) Observes a variable
   B) Intervenes to set a variable
   C) Correlates variables
   D) Conditions on variables

3. **The do-calculus:**
   A) Is always applicable
   B) Provides rules for identification
   C) Requires randomization
   D) Only works for experiments

4. **Counterfactuals:**
   A) Are always observable
   B) Are computed from SCMs
   C) Require experiments
   D) Don't exist

5. **Graph manipulation for do(X):**
   A) Adds edges into X
   B) Removes edges into X
   C) Removes edges out of X
   D) Has no effect

### Short Answer

6. **Explain the difference between $P(Y | X)$ and $P(Y | do(X))$.**

7. **How do you compute a counterfactual?**

8. **What does the do-calculus allow us to do?**

9. **Give an example where $P(Y | do(X))$ differs from $P(Y | X)$.**

10. **How do SCMs relate to DAGs?**

---

## Watch and Connect

```{admonition} Recommended lecture
:class: tip

[**Structural causal models and backdoor adjustment**](https://www.youtube.com/watch?v=dB8r4Afmobo&list=PLoazKTcS0Rzb6bb9L508cyJ1z-U9iWkA0&index=28)

Connects structural assignments, interventions, and graph-based identification.

**Active-viewing prompt:** For each equation, identify what changes under an intervention and what remains invariant.
```

---

## 17.7 Summary

1. **SCMs** formalize causal reasoning with functions
2. **The do-operator** represents interventions
3. **The do-calculus** provides identification rules
4. **Counterfactuals** are computed from SCMs
5. **Graph structure** determines what can be identified

---

## 17.8 Further Reading

- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*. Cambridge University Press.
- Pearl, J. & Mackenzie, D. (2018). *The Book of Why*. Basic Books.


---

## Worked Examples

### Example 1: Pearl's Do-Calculus

Using the three rules of do-calculus to derive causal effects from observational data when certain edges are missing from the graph. The rules allow us to convert interventional distributions to observational distributions under specific conditions.

### Example 2: Front-Door Criterion

When unmeasured confounding blocks the back-door path, but a mediator on the causal path is observed, the front-door criterion identifies the causal effect. This requires that all causal paths from treatment to outcome go through the mediator, and no back-door paths from treatment to mediator exist.

### Example 3: Transportability

Can we transport causal findings from one population to another? SCM provides rules for when this is valid. If the causal structure is the same but the distribution of confounders differs, transport is possible under specific conditions.

### Example 4: Counterfactual Reasoning

"Would the patient have survived if they had received a different treatment?" SCMs formalize this question by defining counterfactual distributions through structural equations.

---

## Diagnostics: SCM Identification

### Identification Algorithm

```python
import networkx as nx

class CausalDAG:
    """Causal Directed Acyclic Graph."""

    def __init__(self):
        self.graph = nx.DiGraph()

    def add_edge(self, parent, child):
        self.graph.add_edge(parent, child)

    def ancestors(self, node):
        return nx.ancestors(self.graph, node)

    def descendants(self, node):
        return nx.descendants(self.graph, node)

    def parents(self, node):
        return list(self.graph.predecessors(node))

    def children(self, node):
        return list(self.graph.successors(node))

    def do(self, node):
        """Simulate intervention do(X=x) by removing incoming edges."""
        graph_copy = self.graph.copy()
        for parent in list(graph_copy.predecessors(node)):
            graph_copy.remove_edge(parent, node)
        return graph_copy

    @staticmethod
    def _is_d_separator(graph, x, y, z):
        """Call the d-separation API across supported NetworkX versions."""
        check_separator = getattr(nx, 'is_d_separator', None)
        if check_separator is None:  # NetworkX versions before 3.5
            check_separator = getattr(nx, 'd_separated')
        return check_separator(graph, {x}, {y}, set(z))

    def is_d_separated(self, x, y, z):
        """Check if X and Y are d-separated given Z."""
        return self._is_d_separator(self.graph, x, y, z)

    def has_unblocked_backdoor(self, treatment, outcome, conditioned_on=()):
        """Check for an open back-door path after conditioning."""
        backdoor_graph = self.graph.copy()
        backdoor_graph.remove_edges_from(list(backdoor_graph.out_edges(treatment)))
        return not self._is_d_separator(
            backdoor_graph, treatment, outcome, conditioned_on
        )

    def front_door_criterion(self, treatment, outcome, mediator):
        """Check if front-door criterion is satisfied."""
        if mediator not in self.graph or mediator == treatment or mediator == outcome:
            return False

        # 1. Every directed treatment-to-outcome path passes through mediator.
        without_mediator = self.graph.copy()
        without_mediator.remove_node(mediator)
        intercepts_all_directed_paths = not nx.has_path(
            without_mediator, treatment, outcome
        )

        # 2. Treatment and mediator have no open back-door path.
        no_treatment_mediator_confounding = not self.has_unblocked_backdoor(
            treatment, mediator
        )

        # 3. Conditioning on treatment blocks mediator-outcome back-door paths.
        mediator_backdoor_graph = self.graph.copy()
        mediator_backdoor_graph.remove_edges_from(
            list(mediator_backdoor_graph.out_edges(mediator))
        )
        mediator_outcome_blocked = self._is_d_separator(
            mediator_backdoor_graph, mediator, outcome, {treatment}
        )

        return (
            intercepts_all_directed_paths
            and no_treatment_mediator_confounding
            and mediator_outcome_blocked
        )
```

### The Three Rules of Do-Calculus

1. **Insert or delete observations**: $P(y \mid do(x), z, w) = P(y \mid do(x), w)$ when $Y$ and $Z$ are d-separated given $X,W$ in $G_{\bar X}$.
2. **Exchange actions and observations**: $P(y \mid do(x), do(z), w) = P(y \mid do(x), z, w)$ when $Y$ and $Z$ are d-separated given $X,W$ in $G_{\bar X,\underline Z}$.
3. **Insert or delete actions**: $P(y \mid do(x), do(z), w) = P(y \mid do(x), w)$ when $Y$ and $Z$ are d-separated given $X,W$ in the appropriately mutilated graph $G_{\bar X,\bar{Z(W)}}$.

Here, a bar removes incoming arrows and an underline removes outgoing arrows. The third rule's $Z(W)$ notation excludes variables in $Z$ that are ancestors of $W$ after incoming arrows to $X$ are removed.

---

## Interpretation Workshop

### Reading SCM Papers

Key concepts:
- **Identification**: Can the causal effect be computed from observational data?
- **do(X)**: Intervention that sets X to a specific value, removing all incoming arrows
- **Counterfactual**: What would have happened under a different treatment?
- **Transportability**: Can findings be generalized to new populations?

### When SCMs Are Useful

- Complex causal structures with multiple pathways
- Questions about what would have happened (counterfactuals)
- Combining experimental and observational data
- Reasoning about interventions in systems

---

## Practical Application

### Implementing Do-Calculus

```python
def identify_effect(dag, treatment, outcome):
    """Attempt to identify causal effect using back-door criterion."""

    # An empty adjustment set is sufficient when no back-door path is open.
    if not dag.has_unblocked_backdoor(treatment, outcome):
        return "No open back-door path — no adjustment required"

    # Check front-door criterion
    mediators = [node for node in dag.descendants(treatment)
                 if outcome in dag.descendants(node)]

    for m in mediators:
        if dag.front_door_criterion(treatment, outcome, m):
            return f"Front-door criterion via {m}"

    return "Not identified with standard criteria"
```

---

## Limitations

- Requires full causal graph: Must know all causal relationships
- Unmeasured confounders: Cannot handle them without additional assumptions
- Complexity: Do-calculus can be difficult to apply in practice
- Data requirements: Need sufficient data to estimate all conditional distributions

---

## Exercises

1. **Graph construction**: Draw the causal graph for a medical treatment scenario. Identify confounders, mediators, and colliders.
2. **d-separation**: Given a graph, determine which variables are d-separated given a set of conditioning variables.
3. **Do-calculus**: Apply the three rules of do-calculus to identify a causal effect in a graph with unmeasured confounding.
4. **Front-door criterion**: Find a scenario where the front-door criterion applies and derive the causal effect formula.

---

## Projects

### Project 1: SCM Implementation
Build a Python class for causal DAGs that can construct graphs, check d-separation, apply do-calculus, and identify causal effects.

### Project 2: Transportability
Implement transportability analysis for moving causal findings from one population to another.


---


---

## Companion Practice

Move from reading to doing with the complete course materials:

- [Download the worked notebook](../notebooks/17-structural-causal-models.ipynb)
- [Download the practice lab](../labs/lab17-structural-causal-models-practice.ipynb)
- [Download the lab solution](../solutions/lab17-solutions.ipynb) — attempt the lab before opening this

The worked notebook demonstrates the lesson's core estimator. The practice lab asks you to make the identification and diagnostic choices yourself.
