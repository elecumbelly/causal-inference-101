#!/usr/bin/env python3
"""Generate synthetic datasets for all Causal Inference 101 lessons."""

import os
import numpy as np
import pandas as pd

np.random.seed(42)

DATASETS_DIR = 'datasets'

def save_csv(df, filename):
    path = os.path.join(DATASETS_DIR, filename)
    df.to_csv(path, index=False)
    print(f"Created: {filename} ({len(df)} rows)")

def lesson_01():
    n = 500
    treatment = np.random.binomial(1, 0.5, n)
    outcome = 2.5 * treatment + np.random.normal(0, 1, n)
    confounder = np.random.normal(0, 1, n)
    save_csv(pd.DataFrame({'treatment': treatment, 'outcome': outcome, 'confounder': confounder}),
             '01-why-causality.csv')

def lesson_02():
    n = 1000
    treatment = np.random.binomial(1, 0.5, n)
    y0 = np.random.normal(0, 1, n)
    y1 = y0 + 2.0 + np.random.normal(0, 0.5, n)
    observed_y = np.where(treatment, y1, y0)
    save_csv(pd.DataFrame({'treatment': treatment, 'Y0': y0, 'Y1': y1, 'observed_outcome': observed_y}),
             '02-potential-outcomes.csv')

def lesson_03():
    n = 300
    x = np.random.normal(0, 1, n)
    m = 0.5 * x + np.random.normal(0, 0.5, n)
    y = 0.3 * x + 0.7 * m + np.random.normal(0, 0.5, n)
    save_csv(pd.DataFrame({'X': x, 'M': m, 'Y': y}), '03-dags.csv')

def lesson_04():
    n = 800
    confounder = np.random.normal(0, 1, n)
    treatment = (confounder + np.random.normal(0, 0.5, n)) > 0
    outcome = 1.5 * treatment + 2.0 * confounder + np.random.normal(0, 1, n)
    save_csv(pd.DataFrame({'confounder': confounder, 'treatment': treatment.astype(int), 'outcome': outcome}),
             '04-confounding.csv')

def lesson_05():
    n = 600
    treatment = np.random.binomial(1, 0.5, n)
    potential_outcome = np.random.normal(5, 2, n)
    observed_outcome = potential_outcome + 3.0 * treatment + np.random.normal(0, 0.5, n)
    save_csv(pd.DataFrame({'treatment': treatment, 'baseline_outcome': potential_outcome, 'observed_outcome': observed_outcome}),
             '05-rct.csv')

def lesson_06():
    n = 1000
    latent = np.random.normal(0, 1, n)
    treatment = (latent + np.random.normal(0, 0.3, n)) > 0
    outcome = 2.0 * treatment + latent + np.random.normal(0, 0.5, n)
    selected = (outcome > np.percentile(outcome, 20))
    save_csv(pd.DataFrame({'treatment': treatment.astype(int), 'outcome': outcome,
                           'latent': latent, 'selected': selected.astype(int)}),
             '06-selection-bias.csv')

def lesson_07():
    n = 500
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    treatment = (0.5 * x1 + 0.3 * x2 + np.random.normal(0, 0.5, n)) > 0
    outcome = 2.0 * treatment + 1.5 * x1 + 0.8 * x2 + np.random.normal(0, 1, n)
    save_csv(pd.DataFrame({'x1': x1, 'x2': x2, 'treatment': treatment.astype(int), 'outcome': outcome}),
             '07-regression.csv')

def lesson_08():
    n = 1200
    age = np.random.uniform(18, 70, n)
    income = 30000 + 500 * age + np.random.normal(0, 10000, n)
    treatment_prob = 1 / (1 + np.exp(-(0.02 * age + 0.00001 * income - 2)))
    treatment = np.random.binomial(1, treatment_prob)
    outcome = 5000 * treatment + 200 * age + 0.1 * income + np.random.normal(0, 5000, n)
    save_csv(pd.DataFrame({'age': age, 'income': income, 'treatment': treatment, 'outcome': outcome}),
             '08-propensity-scores.csv')

def lesson_09():
    n = 800
    instrument = np.random.binomial(1, 0.5, n)
    treatment = 0.7 * instrument + np.random.normal(0, 0.5, n)
    outcome = 3.0 * treatment + np.random.normal(0, 1, n)
    save_csv(pd.DataFrame({'instrument': instrument, 'treatment': treatment, 'outcome': outcome}),
             '09-instrumental-variables.csv')

def lesson_10():
    n_units = 200
    n_periods = 10
    unit_id = np.repeat(range(n_units), n_periods)
    time = np.tile(range(n_periods), n_units)
    treated = (unit_id >= n_units // 2)
    post = (time >= 5)
    effect = treated * post * 2.5
    outcome = 10 + effect + np.random.normal(0, 1, n_units * n_periods)
    save_csv(pd.DataFrame({'unit_id': unit_id, 'time': time, 'treated': treated.astype(int),
                           'post': post.astype(int), 'outcome': outcome}),
             '10-difference-in-differences.csv')

def lesson_11():
    n = 800
    running_var = np.random.uniform(-1, 1, n)
    assignment = (running_var >= 0)
    outcome = 1.0 + 4.0 * assignment + 2.0 * running_var + np.random.normal(0, 1, n)
    save_csv(pd.DataFrame({'running_variable': running_var, 'assignment': assignment.astype(int), 'outcome': outcome}),
             '11-regression-discontinuity.csv')

def lesson_12():
    n_periods = 20
    treated_unit = np.zeros(n_periods) + np.cumsum(np.random.normal(0.1, 0.05, n_periods))
    donor1 = np.cumsum(np.random.normal(0.1, 0.05, n_periods))
    donor2 = np.cumsum(np.random.normal(0.1, 0.05, n_periods))
    donor3 = np.cumsum(np.random.normal(0.1, 0.05, n_periods))
    treated_unit[10:] += 3.0
    save_csv(pd.DataFrame({'period': range(n_periods), 'treated': treated_unit,
                           'donor1': donor1, 'donor2': donor2, 'donor3': donor3}),
             '12-synthetic-control.csv')

def lesson_13():
    n = 600
    treatment = np.random.binomial(1, 0.5, n)
    mediator = 1.5 * treatment + np.random.normal(0, 1, n)
    outcome = 0.8 * mediator + 1.0 * treatment + np.random.normal(0, 0.5, n)
    save_csv(pd.DataFrame({'treatment': treatment, 'mediator': mediator, 'outcome': outcome}),
             '13-mediation.csv')

def lesson_14():
    n = 1000
    x = np.random.normal(0, 1, n)
    treatment = np.random.binomial(1, 0.5, n)
    cate = 1.0 + 2.0 * x
    outcome = cate * treatment + np.random.normal(0, 1, n)
    save_csv(pd.DataFrame({'covariate': x, 'treatment': treatment, 'treatment_effect': cate, 'outcome': outcome}),
             '14-heterogeneous-effects.csv')

def lesson_15():
    n = 500
    treatment = np.random.binomial(1, 0.5, n)
    outcome = 2.0 * treatment + np.random.normal(0, 1, n)
    sensitivity_param = np.random.uniform(0, 0.5, n)
    save_csv(pd.DataFrame({'treatment': treatment, 'outcome': outcome, 'sensitivity_parameter': sensitivity_param}),
             '15-sensitivity-analysis.csv')

def lesson_16():
    n_units = 150
    n_periods = 8
    unit_id = np.repeat(range(n_units), n_periods)
    time = np.tile(range(n_periods), n_units)
    treatment = ((unit_id >= n_units // 2) & (time >= 4)).astype(int)
    outcome = 5 + 0.5 * time + 2.0 * treatment + np.random.normal(0, 1, n_units * n_periods)
    save_csv(pd.DataFrame({'unit_id': unit_id, 'time': time, 'treatment': treatment, 'outcome': outcome}),
             '16-longitudinal.csv')

def lesson_17():
    n = 500
    x = np.random.normal(0, 1, n)
    m = 0.6 * x + np.random.normal(0, 0.5, n)
    y = 0.4 * x + 0.8 * m + np.random.normal(0, 0.5, n)
    save_csv(pd.DataFrame({'X': x, 'M': m, 'Y': y}), '17-structural-causal-models.csv')

def lesson_18():
    n = 400
    a = np.random.normal(0, 1, n)
    b = 0.5 * a + np.random.normal(0, 0.5, n)
    c = 0.3 * a + 0.4 * b + np.random.normal(0, 0.5, n)
    d = 0.6 * b + np.random.normal(0, 0.5, n)
    save_csv(pd.DataFrame({'A': a, 'B': b, 'C': c, 'D': d}), '18-causal-discovery.csv')

def lesson_19():
    n = 800
    x = np.random.normal(0, 1, n)
    treatment = np.random.binomial(1, 1 / (1 + np.exp(-x)))
    outcome = 2.0 * treatment + x + np.random.normal(0, 0.5, n)
    save_csv(pd.DataFrame({'feature': x, 'treatment': treatment, 'outcome': outcome}),
             '19-causal-ml.csv')

def lesson_20():
    n = 500
    treatment = np.random.binomial(1, 0.5, n)
    outcome = 2.5 * treatment + np.random.normal(0, 1, n)
    prior_mean = np.random.normal(0, 0.5, n)
    save_csv(pd.DataFrame({'treatment': treatment, 'outcome': outcome, 'prior_mean': prior_mean}),
             '20-bayesian-causal.csv')

def lesson_21():
    n_source = 500
    n_target = 300
    source_treatment = np.random.binomial(1, 0.5, n_source)
    source_outcome = 2.0 * source_treatment + np.random.normal(0, 1, n_source)
    target_treatment = np.random.binomial(1, 0.3, n_target)
    target_outcome = 1.5 * target_treatment + np.random.normal(0, 1.5, n_target)
    source_df = pd.DataFrame({'group': 'source', 'treatment': source_treatment, 'outcome': source_outcome})
    target_df = pd.DataFrame({'group': 'target', 'treatment': target_treatment, 'outcome': target_outcome})
    save_csv(pd.concat([source_df, target_df], ignore_index=True), '21-external-validity.csv')

def lesson_22():
    n = 600
    state = np.random.choice(['healthy', 'mild', 'severe'], n)
    treatment = np.random.binomial(1, 0.5, n)
    benefit = np.where(state == 'severe', 5.0, np.where(state == 'mild', 2.0, 0.5))
    outcome = benefit * treatment + np.random.normal(0, 1, n)
    cost = 1000 * treatment
    save_csv(pd.DataFrame({'state': state, 'treatment': treatment, 'outcome': outcome, 'cost': cost}),
             '22-decision-theory.csv')

def lesson_23():
    n = 1000
    gender = np.random.binomial(1, 0.5, n)
    treatment = np.random.binomial(1, 0.5, n)
    outcome = 2.0 * treatment + np.random.normal(0, 1, n)
    protected = (gender == 1)
    save_csv(pd.DataFrame({'gender': gender, 'treatment': treatment, 'outcome': outcome, 'protected': protected.astype(int)}),
             '23-fairness.csv')

def lesson_24():
    n = 500
    treatment = np.random.binomial(1, 0.5, n)
    mediator = 1.2 * treatment + np.random.normal(0, 1, n)
    outcome = 0.9 * mediator + 1.5 * treatment + np.random.normal(0, 1, n)
    save_csv(pd.DataFrame({'treatment': treatment, 'mediator': mediator, 'outcome': outcome}),
             '24-capstone.csv')

if __name__ == '__main__':
    os.makedirs(DATASETS_DIR, exist_ok=True)

    generators = [
        lesson_01, lesson_02, lesson_03, lesson_04, lesson_05, lesson_06,
        lesson_07, lesson_08, lesson_09, lesson_10, lesson_11, lesson_12,
        lesson_13, lesson_14, lesson_15, lesson_16, lesson_17, lesson_18,
        lesson_19, lesson_20, lesson_21, lesson_22, lesson_23, lesson_24
    ]

    for gen in generators:
        gen()

    print(f"\nAll {len(generators)} datasets generated in {DATASETS_DIR}/")
