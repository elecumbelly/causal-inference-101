---
title: "Lesson 18: Causal Discovery"
subtitle: "From First Principles to Modern Causal Inference"
---

# Lesson 18: Causal Discovery

## Opening Story: Finding Hidden Causes

In genetics, researchers want to understand which genes cause diseases. They have data on gene expression levels but not the causal structure. Causal discovery methods attempt to learn the causal graph from observational data alone.

This is one of the most challenging problems in causal inference: can we discover causes from correlations?

---

## Learning Objectives

By the end of this lesson, you should be able to:

1. Explain when causal discovery is possible
2. Implement the PC algorithm
3. Understand faithfulness and other assumptions
4. Interpret causal graphs
5. Recognize limitations of causal discovery

---

## 18.1 When Is Causal Discovery Possible?

### Faithfulness Assumption

The causal graph is faithful to the distribution: all conditional independencies in the distribution are represented in the graph.

### Key Results

1. **With faithfulness**: Many conditional independencies can be detected
2. **Without faithfulness**: Causal discovery is generally impossible

### What Can Be Learned

- Undirected skeleton of the graph
- Some v-structures (colliders)
- Partial ordering of variables

---

## 18.2 The PC Algorithm

```python
import numpy as np
import pandas as pd
from causalnex.structure import StructureModel
from causalnex.structure.notears import from_pandas

np.random.seed(42)
n = 1000

# Generate data with known causal structure
Z = np.random.normal(0, 1, n)
X = 0.5 * Z + np.random.normal(0, 0.5, n)
Y = 0.5 * X + 0.3 * Z + np.random.normal(0, 0.5, n)

# Create DataFrame
df = pd.DataFrame({'Z': Z, 'X': X, 'Y': Y})

# NOTEARS algorithm (continuous optimization for structure learning)
sm = from_pandas(df, beta=0.1)
print("Learned edges:")
print(sm.edges)
```

---

## 18.3 Limitations

1. **Faithfulness violations**: Many causal structures are observationally equivalent
2. **Finite sample issues**: Conditional independence tests have low power
3. **Latent variables**: Unmeasured variables complicate discovery
4. **Nonlinearities**: Most methods assume linearity

---

## 18.4 Knowledge Check

### Multiple Choice

1. **Causal discovery requires:**
   A) Randomization
   B) Faithfulness
   C) Large samples
   D) All of the above

2. **The PC algorithm:**
   A) Learns the causal graph
   B) Learns the causal effect
   C) Tests causal hypotheses
   D) All of the above

3. **Faithfulness assumes:**
   A) All independencies are in the graph
   B) All correlations are causal
   C) No confounding
   D) Randomization

4. **Observational equivalence means:**
   A) Different graphs give the same distribution
   B) Observational data is useless
   C) Causal discovery is impossible
   D) Experiments are always needed

5. **Latent variables:**
   A) Help causal discovery
   B) Complicate causal discovery
   C) Have no effect
   D) Are always measured

### Short Answer

6. **Explain the faithfulness assumption.**

7. **What can and cannot be learned from observational data?**

8. **How do latent variables affect causal discovery?**

9. **What role do experiments play in causal discovery?**

10. **Give an example where causal discovery might fail.**

---

## Watch and Connect

```{admonition} Recommended lecture
:class: tip

[**Causal discovery from observational data**](https://www.youtube.com/watch?v=lVE-4deFe7c&list=PLoazKTcS0Rzb6bb9L508cyJ1z-U9iWkA0&index=62)

Explains what discovery algorithms can recover only under Markov, faithfulness, and measurement assumptions.

**Active-viewing prompt:** List which arrow directions remain unidentified in a Markov-equivalence class.
```

---

## 18.5 Summary

1. **Causal discovery** learns causal structure from data
2. **Faithfulness** is a key assumption
3. **PC algorithm** is a classic method
4. **Limitations** include faithfulness violations and latent variables
5. **Experiments** are often needed to confirm causal structure

---

## 18.6 Further Reading

- Spirtes, P., Glymour, C., & Scheines, R. (2000). *Causation, Prediction, and Search*. MIT Press.
- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*. Cambridge University Press.


---

## Worked Examples

### Example 1: PC Algorithm

The PC algorithm discovers causal structure by:
1. Learning the skeleton (undirected graph) through conditional independence tests
2. Identifying v-structures (colliders) from the skeleton
3. Orienting edges based on the v-structures and other constraints

### Example 2: FCI Algorithm

The FCI algorithm handles unmeasured confounders by learning a partial ancestral graph that represents possible latent variables and their relationships to observed variables.

### Example 3: GES Algorithm

The GES algorithm uses score-based optimization:
1. Start with an empty graph
2. Add edges that improve fit most (greedy search)
3. Remove edges that do not improve fit
4. Return the highest-scoring graph using BIC or similar criterion

### Example 4: PCMCI for Time Series

PCMCI discovers causal structure in time series data:
1. Condition on past values to remove temporal confounding
2. Use PC-stable for conditional independence testing
3. Identify contemporaneous and lagged effects

---

## Diagnostics: Discovery Assumptions

### Conditional Independence Testing

```python
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression

def partial_correlation_test(x, y, z, alpha=0.05):
    """Test conditional independence X _||_ Y | Z."""

    # Regress X on Z
    reg_x = LinearRegression().fit(z, x)
    res_x = x - reg_x.predict(z)

    # Regress Y on Z
    reg_y = LinearRegression().fit(z, y)
    res_y = y - reg_y.predict(z)

    # Test correlation of residuals
    r, p_value = stats.pearsonr(res_x, res_y)

    independent = p_value > alpha

    return {
        'partial_corr': r,
        'p_value': p_value,
        'independent': independent
    }
```

### PC Algorithm Implementation

```python
def pc_algorithm(data, alpha=0.05, max_conditioning=3):
    """Simplified PC algorithm for causal discovery."""

    n_vars = data.shape[1]

    # Step 1: Start with complete undirected graph
    adj_matrix = np.ones((n_vars, n_vars)) - np.eye(n_vars)

    # Step 2: Remove edges based on conditional independence
    for cond_size in range(max_conditioning + 1):
        for i in range(n_vars):
            for j in range(n_vars):
                if adj_matrix[i, j] == 1:
                    neighbors = [k for k in range(n_vars)
                                if adj_matrix[i, k] == 1 and k != j]

                    if len(neighbors) >= cond_size:
                        result = partial_correlation_test(
                            data[:, i], data[:, j],
                            data[:, neighbors[:cond_size]],
                            alpha
                        )

                        if result['independent']:
                            adj_matrix[i, j] = 0
                            adj_matrix[j, i] = 0

    return adj_matrix
```

---

## Interpretation Workshop

### Reading Causal Discovery Papers

Key questions:
1. What assumptions are made? (Acyclicity, faithfulness, etc.)
2. What can be identified? (Skeleton, v-structures, full orientation?)
3. How is statistical uncertainty handled? (Multiple testing, confidence)
4. What is the sample size requirement? (Curse of dimensionality)

### Limitations of Causal Discovery

- Faithfulness: May not hold in practice
- Sample size: Requires exponential data for many variables
- Latent confounders: Cannot fully resolve without additional assumptions
- Equivalence classes: Multiple graphs can generate the same distribution

---

## Practical Application

### Causal Discovery with DoWhy

```python
import dowhy
from dowhy import CausalModel

def causal_discovery_dowhy(data, treatment, outcome):
    """Use DoWhy for causal discovery."""

    model = CausalModel(
        data=data,
        treatment=treatment,
        outcome=outcome,
        common_causes=['confounder1', 'confounder2']
    )

    identified = model.identify_effect()
    estimate = model.estimate_effect(identified,
                                      method_name="backdoor.linear_regression")

    return estimate
```

---

## Limitations

- Equivalence classes: Multiple DAGs can be Markov equivalent
- Latent variables: Most algorithms assume no unmeasured confounders
- Sample complexity: Exponential in number of variables
- Faithfulness violations: Can lead to incorrect conclusions

---

## Exercises

1. **Conditional independence**: Implement the conditional independence test. Apply it to a dataset with known structure.
2. **PC algorithm**: Run the PC algorithm on simulated data. How well does it recover the true graph?
3. **Equivalence classes**: Find two different DAGs that are Markov equivalent. Show they generate the same distribution.
4. **Critique**: A paper claims to have discovered causal structure from observational data. What are the threats to validity?

---

## Projects

### Project 1: Causal Discovery Comparison
Compare PC, FCI, and GES algorithms on simulated data with known ground truth.

### Project 2: Causal Discovery with Latent Variables
Implement an algorithm that handles latent confounders (e.g., FCI).


---

## Additional Advanced Content

### PC Algorithm Details

The PC algorithm proceeds in three main phases:

**Phase 1: Skeleton Learning**
- Start with a complete undirected graph
- For each pair of variables (X, Y), test conditional independence X _||_ Y | S for increasing conditioning sets S
- Remove edges where independence is detected
- Record the conditioning sets where edges were removed

**Phase 2: V-Structure Identification**
- For each triple X — Z — Y where Z is not adjacent to X or Y
- Check if Z is in the conditioning set where X and Y were found independent
- If not, orient as X → Z ← Y (v-structure)

**Phase 3: Edge Orientation**
- Apply Meek's rules to orient remaining edges without creating new v-structures or cycles

```python
def pc_algorithm_detailed(data, alpha=0.05, max_conditioning=3):
    """Detailed PC algorithm implementation."""

    n_vars = data.shape[1]
    var_names = [f"X{i}" for i in range(n_vars)]

    # Phase 1: Skeleton
    adj_matrix = np.ones((n_vars, n_vars)) - np.eye(n_vars)
    sep_set = {}  # Store separation sets

    for cond_size in range(max_conditioning + 1):
        for i in range(n_vars):
            for j in range(i+1, n_vars):
                if adj_matrix[i, j] == 1:
                    neighbors = [k for k in range(n_vars)
                                if adj_matrix[i, k] == 1 and k != j]

                    found_independent = False
                    for subset_size in range(cond_size + 1):
                        if subset_size > len(neighbors):
                            break

                        from itertools import combinations
                        for subset in combinations(neighbors, subset_size):
                            result = partial_correlation_test(
                                data[:, i], data[:, j],
                                data[:, list(subset)],
                                alpha
                            )

                            if result['independent']:
                                adj_matrix[i, j] = 0
                                adj_matrix[j, i] = 0
                                sep_set[(i, j)] = list(subset)
                                found_independent = True
                                break

                        if found_independent:
                            break

    # Phase 2: V-structures
    v_structures = []

    for z in range(n_vars):
        neighbors_z = [k for k in range(n_vars) if adj_matrix[z, k] == 1]

        for i in range(len(neighbors_z)):
            for j in range(i+1, len(neighbors_z)):
                x, y = neighbors_z[i], neighbors_z[j]

                # Check if x and y are not adjacent
                if adj_matrix[x, y] == 0:
                    # Check if z is NOT in the separation set of x and y
                    if (x, y) in sep_set and z not in sep_set[(x, y)]:
                        v_structures.append((x, z, y))

    print(f"Detected {len(v_structures)} v-structures")

    return adj_matrix, v_structures
```

### Faithfulness Assumption

The faithfulness assumption states that all conditional independencies in the distribution are reflected in the graph. Violations occur when:

1. **Cancellation**: Two paths between variables cancel each other out
2. **Exact balance**: Opposing effects perfectly offset

```python
def demonstrate_faithfulness_violation(n=10000):
    """Show when faithfulness can be violated."""

    # Generate data where faithfulness is violated
    U = np.random.normal(size=n)
    X = U + np.random.normal(0, 0.5, size=n)
    Y = 0.5*X + 0.5*U + np.random.normal(0, 0.5, size=n)

    # X and Y are dependent (correlated)
    # But conditioning on U might make them appear independent

    corr_xy = np.corrcoef(X, Y)[0, 1]

    # Condition on U
    u_median = np.median(U)
    low_u = U < u_median
    high_u = U >= u_median

    corr_low = np.corrcoef(X[low_u], Y[low_u])[0, 1]
    corr_high = np.corrcoef(X[high_u], Y[high_u])[0, 1]

    print(f"Marginal correlation X-Y: {corr_xy:.3f}")
    print(f"Conditional correlation (U low): {corr_low:.3f}")
    print(f"Conditional correlation (U high): {corr_high:.3f}")
```

### Application to Genomics

Causal discovery in genomics faces unique challenges:
- **High dimensionality**: Thousands of genes, limited samples
- **Feedback loops**: Gene regulatory networks have cycles
- **Latent variables**: Unmeasured transcription factors

```python
def genomic_causal_discovery(expression_data, gene_names, alpha=0.01):
    """Simplified causal discovery for gene expression."""

    n_genes = expression_data.shape[1]

    # Use PC algorithm with conservative alpha
    adj_matrix, v_structures = pc_algorithm_detailed(
        expression_data, alpha=alpha, max_conditioning=2
    )

    # Report results
    print(f"\nCausal Discovery Results:")
    print(f"Genes analyzed: {n_genes}")
    print(f"Edges discovered: {int(np.sum(adj_matrix) / 2)}")
    print(f"V-structures: {len(v_structures)}")

    # Identify potential regulatory relationships
    for i in range(n_genes):
        for j in range(n_genes):
            if adj_matrix[i, j] == 1:
                print(f"  {gene_names[i]} -> {gene_names[j]}")

    return adj_matrix
```

### Limitations and Alternatives

When faithfulness is violated:
- **PC algorithm may fail**: It relies on conditional independence tests
- **Alternative**: Use score-based methods (GES) that are robust to some violations
- **Alternative**: Use stability selection to identify edges that appear consistently

```python
def stability_selection(data, base_method='pc', n_bootstrap=100, threshold=0.7):
    """Stable edges that appear consistently across bootstrap samples."""

    edge_counts = np.zeros((data.shape[1], data.shape[1]))

    for _ in range(n_bootstrap):
        # Bootstrap sample
        idx = np.random.choice(len(data), len(data), replace=True)
        boot_data = data[idx]

        # Run causal discovery
        adj_matrix, _ = pc_algorithm_detailed(boot_data, alpha=0.05, max_conditioning=2)

        # Count edges
        edge_counts += adj_matrix

    # Normalize
    edge_probs = edge_counts / n_bootstrap

    # Select stable edges
    stable_edges = edge_probs >= threshold

    print(f"Stable edges (threshold={threshold}): {int(np.sum(stable_edges) / 2)}")

    return stable_edges
```


---


---

## Companion Practice

Move from reading to doing with the complete course materials:

- [Download the worked notebook](../notebooks/18-causal-discovery.ipynb)
- [Download the practice lab](../labs/lab18-causal-discovery-practice.ipynb)
- [Download the lab solution](../solutions/lab18-solutions.ipynb) — attempt the lab before opening this

The worked notebook demonstrates the lesson's core estimator. The practice lab asks you to make the identification and diagnostic choices yourself.
