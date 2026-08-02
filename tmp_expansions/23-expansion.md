

---

## Worked Examples

### Example 1: Hiring Algorithms

An AI hiring tool trained on historical data may discriminate against women if past hiring decisions were biased. Fairness-aware causal inference attempts to correct for this while estimating the true effect of qualifications on hiring.

### Example 2: Criminal Justice Risk Assessment

Risk assessment tools used for bail decisions may perpetuate racial disparities. Causal fairness asks: would the prediction change if the defendant's race were different, holding all causally relevant factors constant?

### Example 3: Credit Scoring

Credit scoring models may use variables that are proxies for protected attributes. Fairness constraints ensure that similar applicants receive similar scores regardless of race or gender.

### Example 4: Healthcare Resource Allocation

Allocation algorithms for organ transplants or ICU beds must balance efficiency with equity. Causal fairness ensures that allocation rules do not discriminate based on protected characteristics.

---

## Diagnostics: Fairness Definitions

### Individual Fairness

```python
import numpy as np

def individual_fairness(model, data, protected_attr, distance_metric):
    """Check if similar individuals receive similar predictions."""

    predictions = model.predict(data)

    # Find similar pairs
    n = len(data)
    violations = 0

    for i in range(n):
        for j in range(i+1, n):
            dist = distance_metric(data[i], data[j])
            pred_diff = abs(predictions[i] - predictions[j])

            if pred_diff > dist:
                violations += 1

    print(f"Individual fairness violations: {violations}/{n*(n-1)//2}")

    return violations
```

### Group Fairness Metrics

```python
def group_fairness_metrics(y_true, y_pred, protected):
    """Calculate group fairness metrics."""

    groups = np.unique(protected)

    metrics = {}
    for group in groups:
        mask = protected == group
        metrics[group] = {
            'selection_rate': np.mean(y_pred[mask]),
            'true_positive_rate': np.mean(y_pred[mask & (y_true == 1)]),
            'false_positive_rate': np.mean(y_pred[mask & (y_true == 0)]),
        }

    print("Group Fairness Metrics:")
    for group in groups:
        print(f"  Group {group}:")
        for metric, value in metrics[group].items():
            print(f"    {metric}: {value:.3f}")

    # Disparate impact ratio
    if len(groups) == 2:
        ratio = metrics[groups[0]]['selection_rate'] / metrics[groups[1]]['selection_rate']
        print(f"\n  Disparate Impact Ratio: {ratio:.3f}")
        print(f"  (4/5 rule threshold: 0.8)")

    return metrics
```

### Counterfactual Fairness

```python
def counterfactual_fairness(model, individual, protected_attr, causal_model):
    """Check if prediction would change if protected attribute were different."""

    # Original prediction
    pred_original = model.predict(individual.reshape(1, -1))[0]

    # Counterfactual: change protected attribute
    counterfactual = individual.copy()
    counterfactual_idx = list(causal_model.variables).index(protected_attr)
    counterfactual[counterfactual_idx] = 1 - counterfactual[counterfactual_idx]

    pred_counterfactual = model.predict(counterfactual.reshape(1, -1))[0]

    diff = abs(pred_original - pred_counterfactual)

    print(f"Original prediction: {pred_original:.3f}")
    print(f"Counterfactual prediction: {pred_counterfactual:.3f}")
    print(f"Difference: {diff:.3f}")

    return diff
```

---

## Interpretation Workshop

### Trade-offs Between Fairness Definitions

- **Individual vs group**: Can satisfy group fairness while violating individual fairness
- **Calibration vs balance**: Different fairness criteria may be incompatible
- **Accuracy vs fairness**: Reducing bias may reduce overall accuracy

### When Fairness Analysis Matters

- Automated decisions affect people's lives (hiring, lending, criminal justice)
- Historical data may encode past discrimination
- Protected attributes may correlate with other features
- Stakeholders have different fairness preferences

---

## Practical Application

### Fairness-Aware Machine Learning

```python
def fairness_constrained_training(X, y, protected, constraint_type='demographic_parity'):
    """Train model with fairness constraints."""

    from sklearn.linear_model import LogisticRegression

    # Standard model
    model_standard = LogisticRegression(max_iter=1000)
    model_standard.fit(X, y)

    # Fairness-aware model (simplified)
    # In practice, use libraries like fairlearn or aif360

    # Evaluate fairness
    y_pred = model_standard.predict(X)
    group_fairness_metrics(y, y_pred, protected)

    return model_standard
```

### Post-Processing for Fairness

```python
def post_process_fairness(y_pred, protected, target_rate=0.5):
    """Adjust predictions to achieve fairness."""

    groups = np.unique(protected)
    adjusted = y_pred.copy()

    for group in groups:
        mask = protected == group
        current_rate = np.mean(y_pred[mask])

        if current_rate > target_rate:
            # Randomly flip some positive predictions
            n_flip = int((current_rate - target_rate) * mask.sum())
            flip_idx = np.where(mask & (y_pred == 1))[0][:n_flip]
            adjusted[flip_idx] = 0
        elif current_rate < target_rate:
            # Randomly flip some negative predictions
            n_flip = int((target_rate - current_rate) * mask.sum())
            flip_idx = np.where(mask & (y_pred == 0))[0][:n_flip]
            adjusted[flip_idx] = 1

    return adjusted
```

---

## Limitations

- Multiple incompatible fairness definitions
- Trade-off between fairness and accuracy
- Protected attributes may be unobserved
- Fairness criteria may not capture all ethical concerns

---

## Exercises

1. **Fairness metrics**: Calculate demographic parity, equalized odds, and calibration for a classification model.
2. **Trade-off analysis**: Show that equalized odds and calibration cannot both be satisfied simultaneously in some cases.
3. **Counterfactual fairness**: Implement counterfactual fairness for a simple model.
4. **Policy**: Design a hiring algorithm that satisfies demographic parity. What are the consequences?

---

## Projects

### Project 1: Fairness Audit
Conduct a fairness audit of a real-world algorithm using multiple fairness metrics.

### Project 2: Fairness-Aware Learning
Implement a fairness-aware machine learning algorithm and compare to unconstrained learning.
