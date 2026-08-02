

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
