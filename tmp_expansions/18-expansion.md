

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
