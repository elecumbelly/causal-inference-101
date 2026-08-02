---
title: "Lesson 3: Directed Acyclic Graphs"
subtitle: "From First Principles to Modern Causal Inference"
---

# Lesson 3: Directed Acyclic Graphs

## Opening Story

Suppose a treatment appears beneficial overall but harmful within every severity group. The arithmetic is not contradictory: treatment choice and disease severity may be related, so the aggregate comparison mixes patients with different prognoses. This pattern is known as Simpson's paradox.

A directed acyclic graph, or DAG, does not make the decision automatically. It forces us to state the causal story that determines whether severity is a confounder to adjust for, a mediator to preserve, or a collider to leave alone.

A DAG is a simple picture that shows the causal relationships between variables. Arrows point from causes to effects. By examining the structure of a DAG, we can determine exactly what we need to control for to estimate a causal effect—and what we should *not* control for.

---

## Learning Objectives

By the end of this lesson, you should be able to:

1. Draw and interpret DAGs
2. Identify chains, forks, and colliders
3. Apply d-separation to determine conditional independence
4. Use the backdoor criterion to find adjustment sets
5. Understand the frontdoor criterion
6. Explain why controlling for colliders is harmful

---

```{figure} ../figures/instructional/dag-motifs.svg
:name: lesson-03-dag-motifs
:alt: Conditioning has opposite consequences for chains/forks and colliders.
:width: 100%

Conditioning has opposite consequences for chains/forks and colliders.
```

---

## 3.1 What is a DAG?

### The Basic Idea

A Directed Acyclic Graph (DAG) is a graphical representation of causal relationships where:

- **Nodes** represent variables
- **Directed edges** (arrows) represent direct causal effects
- **Acyclic** means there are no feedback loops (you can't follow arrows and return to where you started)

### Why DAGs Matter

DAGs serve three crucial purposes:

1. **Encode causal assumptions**: They make our assumptions explicit and transparent
2. **Identify what to control for**: They tell us exactly which variables to adjust for
3. **Detect potential problems**: They reveal confounding, colliders, and other issues

### Example: Smoking and Cancer

```python
import networkx as nx
import matplotlib.pyplot as plt

# Create a simple DAG
G = nx.DiGraph()
G.add_edges_from([('Smoking', 'Tar'), ('Tar', 'Cancer')])

# Draw the DAG
plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, ax=plt.gca(), with_labels=True, node_size=3000,
        node_color='lightblue', font_size=12, font_weight='bold',
        arrowsize=20)
plt.title('Simple DAG: Smoking → Tar → Cancer')
plt.show()
```

---

## 3.2 Three Basic Structures

### Chains

A chain is a sequence of causal effects: $X \rightarrow Z \rightarrow Y$

- $X$ causes $Z$, which causes $Y$
- The effect of $X$ on $Y$ flows through $Z$
- If we condition on $Z$, we block the path from $X$ to $Y$

**Example**: Smoking → Tar → Cancer

```python
# Chain
G_chain = nx.DiGraph()
G_chain.add_edges_from([('X', 'Z'), ('Z', 'Y')])

plt.figure(figsize=(6, 4))
pos = {'X': (0, 0), 'Z': (1, 0), 'Y': (2, 0)}
nx.draw(G_chain, pos, ax=plt.gca(), with_labels=True, node_size=3000,
        node_color='lightblue', font_size=14, font_weight='bold',
        arrowsize=20)
plt.title('Chain: X → Z → Y')
plt.show()
```

### Forks

A fork is a common cause: $X \leftarrow Z \rightarrow Y$

- $Z$ causes both $X$ and $Y$
- $X$ and $Y$ are associated because of $Z$ (confounding)
- If we condition on $Z$, we block the spurious association

**Example**: Temperature → Ice Cream Sales, Temperature → Drowning

```python
# Fork (confounding)
G_fork = nx.DiGraph()
G_fork.add_edges_from([('Z', 'X'), ('Z', 'Y')])

plt.figure(figsize=(6, 4))
pos = {'Z': (1, 1), 'X': (0, 0), 'Y': (2, 0)}
nx.draw(G_fork, pos, ax=plt.gca(), with_labels=True, node_size=3000,
        node_color='lightgreen', font_size=14, font_weight='bold',
        arrowsize=20)
plt.title('Fork: X ← Z → Y (Confounding)')
plt.show()
```

### Colliders

A collider is a common effect: $X \rightarrow Z \leftarrow Y$

- Both $X$ and $Y$ cause $Z$
- $X$ and $Y$ are independent (no association)
- If we condition on $Z$, $X$ and $Y$ become associated (Berkson's paradox)

**Example**: Disease A → Hospital, Disease B → Hospital

```python
# Collider
G_collider = nx.DiGraph()
G_collider.add_edges_from([('X', 'Z'), ('Y', 'Z')])

plt.figure(figsize=(6, 4))
pos = {'X': (0, 1), 'Y': (2, 1), 'Z': (1, 0)}
nx.draw(G_collider, pos, ax=plt.gca(), with_labels=True, node_size=3000,
        node_color='lightyellow', font_size=14, font_weight='bold',
        arrowsize=20)
plt.title('Collider: X → Z ← Y')
plt.show()
```

---

## 3.3 D-Separation

### The Rule

Two variables $X$ and $Y$ are **d-separated** given a set of variables $Z$ if every path between $X$ and $Y$ is "blocked" by $Z$.

A path is blocked if:
1. It contains a chain $A \rightarrow B \rightarrow C$ or a fork $A \leftarrow B \rightarrow C$ where $B \in Z$
2. It contains a collider $A \rightarrow B \leftarrow C$ where $B \notin Z$ and no descendant of $B$ is in $Z$

### D-Separation Rules

| Structure | Conditioning | Path Status |
|-----------|--------------|-------------|
| Chain: $X \rightarrow Z \rightarrow Y$ | Condition on $Z$ | Blocked |
| Fork: $X \leftarrow Z \rightarrow Y$ | Condition on $Z$ | Blocked |
| Collider: $X \rightarrow Z \leftarrow Y$ | Don't condition on $Z$ | Blocked |
| Collider: $X \rightarrow Z \leftarrow Y$ | Condition on $Z$ | **Opened** |

### Key Insight

**Conditioning on colliders opens paths that were previously blocked.** This is why controlling for a collider can introduce bias.

---

## 3.4 The Backdoor Criterion

### Definition

A set of variables $Z$ satisfies the **backdoor criterion** relative to $X$ and $Y$ if:
1. No node in $Z$ is a descendant of $X$
2. $Z$ blocks every path between $X$ and $Y$ that contains an arrow into $X$

### Why It Works

The backdoor criterion ensures that we've blocked all confounding paths (paths that create spurious association between $X$ and $Y$).

### Example

In the smoking-cancer DAG:

```
Confounder
   ↓   ↘
Smoking → Tar → Cancer
```

The set {Confounder} satisfies the backdoor criterion because it blocks the backdoor path from Smoking to Cancer through the confounder.

### Finding Adjustment Sets

To estimate the causal effect of $X$ on $Y$:
1. Draw the DAG
2. Identify all backdoor paths from $X$ to $Y$
3. Find a set $Z$ that blocks all backdoor paths
4. Ensure $Z$ doesn't include descendants of $X$
5. Adjust for $Z$ using regression, matching, or weighting

---

## 3.5 The Frontdoor Criterion

### When Backdoor Fails

Sometimes we can't satisfy the backdoor criterion because:
- We can't measure all confounders
- There are unobserved variables on backdoor paths

### Definition

A set of variables $M$ satisfies the **frontdoor criterion** relative to $X$ and $Y$ if:
1. $X$ intercepts all directed paths from $X$ to $Y$
2. There are no unblocked backdoor paths from $X$ to $M$
3. All backdoor paths from $M$ to $Y$ are blocked by $X$

### Example

Consider:

```
U
↓ ↘
X → M → Y
```

If we can't observe $U$, we can't use the backdoor criterion. But $M$ might satisfy the frontdoor criterion:
1. All directed paths from $X$ to $Y$ go through $M$ ✓
2. No backdoor paths from $X$ to $M$ (except through $U$, which is blocked) ✓
3. Backdoor paths from $M$ to $Y$ are blocked by $X$ ✓

### The Frontdoor Adjustment

If $M$ satisfies the frontdoor criterion, we can estimate the causal effect as:

$$P(Y | do(X)) = \sum_m P(M = m | X) \sum_{x'} P(Y | M = m, X = x') P(X = x')$$

---

## 3.6 Minimal Adjustment Sets

### The Principle

We want to control for the **minimum** set of variables needed to block all confounding paths. Controlling for too many variables can introduce bias.

### What NOT to Control For

1. **Colliders**: Conditioning on colliders opens spurious paths
2. **Descendants of treatment**: These are affected by treatment and shouldn't be controlled for
3. **Mediators**: If you want the total effect, don't control for mediators

### The "Do-Calculus"

Judea Pearl developed a complete set of rules (the do-calculus) for determining when causal effects can be estimated from observational data. The backdoor and frontdoor criteria are special cases of these rules.

---

## 3.7 Python Workshop: Drawing and Analyzing DAGs

### Using NetworkX

```python
import networkx as nx
import matplotlib.pyplot as plt

# Create a more complex DAG
G = nx.DiGraph()
G.add_edges_from([
    ('Education', 'Income'),
    ('Ability', 'Education'),
    ('Ability', 'Income'),
    ('Family Background', 'Education'),
    ('Family Background', 'Income')
])

# Draw the DAG
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, ax=plt.gca(), with_labels=True, node_size=3000,
        node_color='lightblue', font_size=12, font_weight='bold',
        arrowsize=20)
plt.title('DAG: Education and Income')
plt.show()
```

### Checking D-Separation

```python
# Check d-separation with NetworkX's tested implementation
def is_dseparated(G, x, y, z):
    """Return whether x and y are d-separated by conditioning set z."""
    check_separator = getattr(nx, 'is_d_separator', None)
    if check_separator is None:  # NetworkX versions before 3.5
        check_separator = getattr(nx, 'd_separated')
    return check_separator(G, {x}, {y}, set(z))

separated = is_dseparated(G, 'Education', 'Income', {'Ability'})
print(f"Education and Income d-separated given Ability? {separated}")
print("No: the causal arrow remains open, as does the path through Family Background.")
```

### Using DAGitty

For more complex DAG analysis, consider using the DAGitty package or web tool:

```python
# DAGitty is primarily a web tool, but you can use the logic here
# For complex DAGs, draw them and check d-separation visually

# Example: The smoking-tar-cancer DAG
G_smoking = nx.DiGraph()
G_smoking.add_edges_from([
    ('Confounder', 'Smoking'),
    ('Confounder', 'Cancer'),
    ('Smoking', 'Tar'),
    ('Tar', 'Cancer')
])

plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G_smoking, seed=42)
nx.draw(G_smoking, pos, ax=plt.gca(), with_labels=True, node_size=3000,
        node_color='lightblue', font_size=12, font_weight='bold',
        arrowsize=20)

# Highlight backdoor path
backdoor_path = [('Smoking', 'Confounder'), ('Confounder', 'Cancer')]
nx.draw_networkx_edges(G_smoking, pos, edgelist=backdoor_path,
                       edge_color='red', width=3, style='dashed')

plt.title('DAG: Smoking, Tar, and Cancer\n(Red = Backdoor Path)')
plt.show()
```

---

## 3.8 Case Study: Smoking, Tar, and Cancer

### The Historical Debate

For decades, the tobacco industry argued that the association between smoking and cancer could be explained by confounding. They suggested that:
- Genetics might cause both smoking and cancer
- Stress might cause both smoking and cancer
- Diet might cause both smoking and cancer

### The DAG Analysis

The causal diagram looks like:

```
Genetics
   ↓   ↘
Smoking → Tar → Cancer
```

The backdoor criterion tells us: we need to control for Genetics (and other confounders) to estimate the causal effect of Smoking on Cancer.

### What Epidemiologists Did

1. **Identified potential confounders**: Genetics, stress, diet, exercise
2. **Measured and controlled for them**: In observational studies
3. **Found the effect persisted**: Even after controlling for confounders
4. **Used multiple methods**: Consistent results across different approaches

### The Bradford Hill Criteria

The Surgeon General's 1964 report used nine criteria to argue for causation:
1. Strength of association
2. Consistency
3. Specificity
4. Temporality
5. Biological gradient
6. Plausibility
7. Coherence
8. Experiment
9. Analogy

---

## 3.9 Common Mistakes

### Mistake 1: Controlling for Colliders

**The error**: Conditioning on a variable that is a common effect of treatment and outcome.

**Why it happens**: People think "more control is better" or confuse colliders with confounders.

**Example**: Controlling for "health status" when studying whether exercise causes heart health. Health status is affected by both exercise and heart health.

### Mistake 2: Controlling for Mediators

**The error**: Controlling for a variable that lies on the causal pathway.

**Why it happens**: People want to "isolate" the direct effect, but this removes part of the causal effect.

**Example**: Controlling for "job performance" when studying whether education causes income. Job performance is a mediator.

### Mistake 3: Drawing the Wrong DAG

**The error**: Making incorrect causal assumptions about the relationships between variables.

**How to avoid it**: Use domain knowledge, prior research, and biological plausibility to inform your DAG.

---

## 3.10 Discussion Questions

1. **Education and Earnings**: Draw a DAG for the relationship between education and earnings. What confounders should you control for? What mediators should you not control for?

2. **Colliders in Hiring**: A company hires based on interview performance. Interview performance is affected by both ability and charisma. If we condition on being hired, what happens to the correlation between ability and charisma?

3. **Frontdoor Example**: Find a real-world example where the frontdoor criterion might be applicable.

4. **Minimal Adjustment**: Why is it better to control for fewer variables rather than more?

5. **DAG Criticism**: What are the limitations of DAGs as a tool for causal inference?

---

## 3.11 Interview Questions

### Question 1
**"What is a collider and why shouldn't we condition on it?"**

**Model Answer**: A collider is a variable that is causally affected by two other variables—both have arrows pointing into it. For example, in a study of exercise and heart health, "health status" is a collider because both exercise and heart health affect it. We shouldn't condition on colliders because doing so creates a spurious association between the two causes. This is called Berkson's paradox or collider bias. Conditioning on a collider opens a path that was previously blocked, making independent variables appear dependent.

### Question 2
**"Explain the backdoor criterion in simple terms."**

**Model Answer**: The backdoor criterion is a rule for determining which variables to control for when estimating a causal effect. It says: find all the "backdoor paths" from treatment to outcome (paths that start with an arrow into treatment), and control for variables that block all those paths without controlling for descendants of treatment. This ensures we're blocking confounding without introducing new biases. The criterion is both necessary and sufficient for identifying the causal effect from observational data.

### Question 3
**"What's the difference between a chain, a fork, and a collider?"**

**Model Answer**: A chain is $X \rightarrow Z \rightarrow Y$—X causes Z, which causes Y. A fork is $X \leftarrow Z \rightarrow Y$—Z causes both X and Y (confounding). A collider is $X \rightarrow Z \leftarrow Y$—both X and Y cause Z. The key difference is how conditioning affects them: conditioning on the middle node in a chain or fork blocks the association, but conditioning on a collider opens a new association.

### Question 4
**"When might the frontdoor criterion be useful?"**

**Model Answer**: The frontdoor criterion is useful when we can't measure all confounders but there's a mediator that satisfies specific conditions. For example, if we want to know whether advertising causes sales, but we can't measure all confounders (like customer preferences), we might use brand awareness as a mediator. If advertising affects sales only through brand awareness, and there are no unmeasured confounders between advertising and brand awareness, we can use the frontdoor criterion to estimate the causal effect.

### Question 5
**"Draw a DAG for the effect of a job training program on earnings. What should you control for?"**

**Model Answer**: The DAG would show: Baseline ability → Training program, Baseline ability → Earnings, Training program → Earnings. We should control for baseline ability (and other confounders like education, age, prior income) to block the backdoor path. We should NOT control for post-training outcomes like "job satisfaction" or "skills gained" because these are mediators. We should also be careful not to control for variables that are affected by the treatment.

---

## 3.12 Knowledge Check

### Multiple Choice

1. **A collider is:**
   - A) A variable that causes both treatment and outcome
   - B) A variable caused by both treatment and outcome
   - C) A variable on the causal pathway
   - D) A variable unrelated to treatment or outcome

2. **Conditioning on a collider:**
   - A) Blocks spurious associations
   - B) Creates spurious associations
   - C) Has no effect
   - D) Depends on the DAG

3. **The backdoor criterion requires:**
   - A) No descendants of treatment in the adjustment set
   - B) Blocking all backdoor paths
   - C) Both A and B
   - D) Neither A nor B

4. **In a chain $X \rightarrow Z \rightarrow Y$, conditioning on Z:**
   - A) Blocks the path from X to Y
   - B) Opens the path from X to Y
   - C) Has no effect
   - D) Depends on other factors

5. **The frontdoor criterion is useful when:**
   - A) All confounders are observed
   - B) There's a mediator satisfying specific conditions
   - C) The treatment is randomly assigned
   - D) There are no colliders

### Short Answer

6. **Explain why controlling for mediators can bias your estimate of the total effect.**

7. **Give an example of a real-world collider.**

8. **What is the difference between d-separation and conditional independence?**

9. **Why should we prefer minimal adjustment sets?**

10. **Draw a DAG showing the relationship between exercise, weight, and heart disease.**

---

## Watch and Connect

```{admonition} Recommended lecture
:class: tip

[**Graphical causal models**](https://www.youtube.com/watch?v=Go4EkHN_PcA&list=PLoazKTcS0Rzb6bb9L508cyJ1z-U9iWkA0&index=19)

A systematic treatment of DAGs, paths, and graphical identification.

**Active-viewing prompt:** For every fork, chain, and collider, predict whether conditioning opens or closes the path.
```

---

## 3.13 Summary

In this lesson, we learned:

1. **DAGs** are graphical representations of causal relationships
2. **Three basic structures**: chains, forks, and colliders
3. **D-separation** determines conditional independence given a DAG
4. **Backdoor criterion** identifies what to control for
5. **Frontdoor criterion** handles unmeasured confounders
6. **Colliders** should not be conditioned on (they create bias)
7. **Minimal adjustment** is preferred—control for less, not more

---

## 3.14 Further Reading

### Classic Works
- Pearl, J. (1995). "Causal Diagrams for Empirical Research." *Biometrika*.
- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*. Cambridge University Press.
- Spirtes, P., Glymour, C., & Scheines, R. (2000). *Causation, Prediction, and Search*. MIT Press.

### Modern Textbooks
- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*. Cambridge University Press.
- Cunningham, S. (2021). *Causal Inference: The Mixtape*. MIT Press.
- Huntington-Klein, N. (2021). *The Effect*. Chapman and Hall/CRC.

### Online Resources
- [DAGitty](https://www.dagitty.net/) - Interactive DAG tool
- [DoWhy: Modeling Causal Relations](https://www.pywhy.org/dowhy/v0.14/user_guide/modeling_causal_relations/index.html) - Causal graphs and identification in Python


---

## Worked Examples

### Example 1: Reading a DAG

In a DAG, arrows represent direct causal effects. If X → Y, X directly causes Y. If X → Z → Y, the effect of X on Y is mediated by Z.

```python
import networkx as nx
import matplotlib.pyplot as plt

# Create a simple DAG
G = nx.DiGraph()
G.add_edges_from([('X', 'Z'), ('Z', 'Y'), ('X', 'Y')])

# Is X a cause of Y?
print(f"X causes Y: {nx.has_path(G, 'X', 'Y')}")
# What about through Z?
print(f"X causes Y through Z: {nx.has_path(G, 'X', 'Y')}")

# Draw the DAG
plt.figure(figsize=(6, 4))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, ax=plt.gca(), with_labels=True, node_size=2000, node_color='lightblue', arrows=True)
plt.title("Simple DAG: X → Z → Y and X → Y")
plt.show()
```

### Example 2: d-Separation and Conditional Independence

Two variables are d-separated given a set Z if all paths between them are blocked. Blocked paths go through chains (→ Z →) or forks (← Z →) where Z is conditioned on, or through colliders (→ Z ←) where Z is NOT conditioned on.

```python
import networkx as nx

# Fork: X ← Z → Y
G1 = nx.DiGraph()
G1.add_edges_from([('Z', 'X'), ('Z', 'Y')])
print("Fork X ← Z → Y:")
print(f"  X _||_ Y | Z: {not nx.has_path(G1, 'X', 'Y')}")

# Chain: X → Z → Y
G2 = nx.DiGraph()
G2.add_edges_from([('X', 'Z'), ('Z', 'Y')])
print("Chain X → Z → Y:")
print(f"  X _||_ Y | Z: {not nx.has_path(G2, 'X', 'Y')}")

# Collider: X → Z ← Y
G3 = nx.DiGraph()
G3.add_edges_from([('X', 'Z'), ('Y', 'Z')])
print("Collider X → Z ← Y:")
print(f"  X _||_ Y: {not nx.has_path(G3, 'X', 'Y')}")
print(f"  X _||_ Y | Z: {nx.has_path(G3, 'X', 'Y')}")  # Conditioning opens the path!
```


---

## Exercises

### Exercise 1

Draw a DAG representing the causal relationships between education, experience, wage, and ability. Identify confounders and colliders.

### Exercise 2

For the DAG X → Z ← Y, explain in your own words why conditioning on Z creates an association between X and Y.

### Exercise 3

Use the networkx library to create and visualize a DAG with at least 5 variables. Write code to test d-separation between two variables.
