

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
