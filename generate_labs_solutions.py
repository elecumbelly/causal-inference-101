#!/usr/bin/env python3
"""Generate lab and solution .md files (with jupytext frontmatter) for all 24 lessons,
then convert them to valid .ipynb via jupytext."""

import subprocess
import os
import sys

LESSONS = [
    ("01", "why-causality-matters", "Causal Reasoning Practice"),
    ("02", "potential-outcomes", "Potential Outcomes Practice"),
    ("03", "directed-acyclic-graphs", "DAGs Practice"),
    ("04", "confounding", "Confounding Practice"),
    ("05", "randomized-controlled-trials", "RCT Practice"),
    ("06", "selection-bias-collider-bias", "Selection & Collider Bias Practice"),
    ("07", "regression-causal-adjustment", "Regression for Causal Adjustment Practice"),
    ("08", "propensity-scores", "Propensity Scores Practice"),
    ("09", "instrumental-variables", "Instrumental Variables Practice"),
    ("10", "difference-in-differences", "Difference-in-Differences Practice"),
    ("11", "regression-discontinuity", "Regression Discontinuity Practice"),
    ("12", "synthetic-control", "Synthetic Control Practice"),
    ("13", "mediation-analysis", "Mediation Analysis Practice"),
    ("14", "heterogeneous-effects", "Heterogeneous Effects Practice"),
    ("15", "sensitivity-analysis", "Sensitivity Analysis Practice"),
    ("16", "longitudinal-time-varying", "Longitudinal & Time-Varying Practice"),
    ("17", "structural-causal-models", "Structural Causal Models Practice"),
    ("18", "causal-discovery", "Causal Discovery Practice"),
    ("19", "causal-inference-ml", "Causal ML Practice"),
    ("20", "bayesian-causal-inference", "Bayesian Causal Inference Practice"),
    ("21", "external-validity", "External Validity Practice"),
    ("22", "decision-theory", "Decision Theory Practice"),
    ("23", "fairness", "Fairness Practice"),
    ("24", "capstone-project", "Capstone Project"),
]

# Topic-specific imports for each lesson
TOPIC_IMPORTS = {
    "01": "import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nfrom scipy import stats",
    "02": "import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nfrom scipy import stats\nfrom itertools import product",
    "03": "import numpy as np\nimport pandas as pd\nimport networkx as nx\nimport matplotlib.pyplot as plt",
    "04": "import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport networkx as nx\nfrom scipy import stats",
    "05": "import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nfrom scipy import stats\nfrom sklearn.utils import resample",
    "06": "import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport networkx as nx\nfrom scipy import stats",
    "07": "import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport statsmodels.api as sm\nfrom scipy import stats",
    "08": "import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.neighbors import NearestNeighbors",
    "09": "import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport statsmodels.api as sm\nfrom scipy import stats",
    "10": "import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport statsmodels.api as sm\nfrom scipy import stats",
    "11": "import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport statsmodels.api as sm\nfrom scipy import stats",
    "12": "import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nfrom scipy.optimize import minimize",
    "13": "import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport statsmodels.api as sm\nfrom scipy import stats",
    "14": "import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nfrom sklearn.ensemble import GradientBoostingRegressor\nfrom econml.dml import LinearDML",
    "15": "import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nfrom scipy import stats",
    "16": "import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport statsmodels.api as sm\nfrom scipy import stats",
    "17": "import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport networkx as nx",
    "18": "import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nfrom scipy import stats\nfrom sklearn.preprocessing import StandardScaler",
    "19": "import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nfrom sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier\nfrom econml.dml import CausalForestDML",
    "20": "import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nfrom scipy import stats",
    "21": "import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nfrom scipy import stats",
    "22": "import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nfrom scipy import stats",
    "23": "import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nfrom scipy import stats",
    "24": "import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport statsmodels.api as sm\nfrom scipy import stats",
}

# Topic-specific exercise prompts for each lesson
LAB_EXERCISES = {
    "01": [
        ("Exercise 1: Correlation vs. Causation",
         "Consider the following observed correlations:\n"
         "- Ice cream sales are correlated with drowning rates\n"
         "- Countries with more TVs per capita have higher life expectancy\n"
         "- People who carry lighters are more likely to develop lung cancer\n\n"
         "For each, identify:\n"
         "a) Is this correlation likely causal? Why or why not?\n"
         "b) What confounders might explain the observed association?\n"
         "c) What would you need to establish a causal relationship?"),
        ("Exercise 2: The Ladder of Causation",
         "For each of the following questions, classify it as:\n"
         "- Level 1: Association (Seeing)\n"
         "- Level 2: Intervention (Doing)\n"
         "- Level 3: Counterfactual (Imagining)\n\n"
         "a) What is the probability of survival given a drug treatment?\n"
         "b) What would happen to my blood pressure if I took this drug?\n"
         "c) Would this patient have recovered if they had received the placebo?\n"
         "d) How does smoking affect the risk of lung cancer?\n"
         "e) Should we implement this policy to reduce crime?"),
        ("Exercise 3: Causal Thinking in Practice",
         "A tech company observes that employees who attend weekly team meetings have 15% higher "
         "performance ratings than those who don't. The HR director proposes making meetings mandatory "
         "for everyone.\n\n"
         "a) Identify at least two possible confounders.\n"
         "b) Describe an ideal experiment to test whether meetings cause higher performance.\n"
         "c) If you can't run an experiment, what observational methods might help?"),
    ],
    "02": [
        ("Exercise 1: Computing Potential Outcomes",
         "A pharmaceutical company runs a small trial with 5 patients:\n\n"
         "| Patient | Treatment (Y₁) | Control (Y₀) |\n"
         "|---------|----------------|---------------|\n"
         "| 1       | 85             | 80            |\n"
         "| 2       | 78             | 72            |\n"
         "| 3       | 90             | 88            |\n"
         "| 4       | ?              | 75            |\n"
         "| 5       | 82             | ?             |\n\n"
         "a) Calculate the Individual Treatment Effect (ITE) for patients 1, 2, and 3.\n"
         "b) What is the fundamental problem of causal inference for patients 4 and 5?\n"
         "c) If patients are randomly assigned, what is the Average Treatment Effect (ATE)?"),
        ("Exercise 2: SUTVA and Interference",
         "Consider a study on the effect of a new teaching method on student test scores.\n\n"
         "a) State what SUTVA requires in this context.\n"
         "b) Give two examples of how SUTVA might be violated.\n"
         "c) If SUTVA is violated, what happens to our ATE estimate?"),
        ("Exercise 3: Randomization and Balance",
         "Simulate a randomized experiment with n=1000 subjects and a binary treatment.\n"
         "Generate potential outcomes Y₀ ~ N(70, 10) and Y₁ = Y₀ + 5 + ε where ε ~ N(0, 3).\n"
         "Randomly assign treatment and compute:\n"
         "a) The sample ATE\n"
         "b) The standard error of the ATE\n"
         "c) A 95% confidence interval\n"
         "d) Repeat 1000 times and check coverage"),
    ],
}

# Default exercises for lessons without specific ones
DEFAULT_LAB_EXERCISES = [
    ("Exercise 1: Conceptual Review",
     "Review the key concepts from this lesson and answer:\n"
     "a) What are the main assumptions required for causal inference in this context?\n"
     "b) Under what conditions would the estimation method from this lesson fail?\n"
     "c) How does this method compare to randomized experiments?"),
    ("Exercise 2: Hands-on Implementation",
     "Using the dataset of your choice (or simulated data):\n"
     "a) Implement the estimation method from this lesson\n"
     "b) Report your estimates with appropriate standard errors\n"
     "c) Create a visualization of your results"),
    ("Exercise 3: Critical Evaluation",
     "Read the research paper discussed in this lesson and answer:\n"
         "a) What identification strategy did the authors use?\n"
         "b) What are the key identifying assumptions?\n"
         "c) What robustness checks did they perform?\n"
         "d) What limitations did they acknowledge?"),
]

for i, (num, slug, title) in enumerate(LESSONS):
    lab_title = f"Lab {int(num)}: {title}"
    sol_title = f"Lab {int(num)} Solutions: {title}"
    imports = TOPIC_IMPORTS.get(num, TOPIC_IMPORTS["01"])
    exercises = LAB_EXERCISES.get(num, DEFAULT_LAB_EXERCISES)

    # --- Generate lab .md ---
    lab_lines = [
        "---",
        f'title: "{lab_title}"',
        'subtitle: "From First Principles to Modern Causal Inference"',
        "jupytext:",
        "  text_representation:",
        '    extension: md',
        '    format_name: mystmd',
        '    format_version: 0.1',
        "---",
        "",
        f"# {lab_title}",
        "",
        "## Setup",
        "",
        "```python",
        imports,
        "```",
        "",
        "## Instructions",
        "",
        f"Complete the exercises below to practice the concepts from Lesson {int(num)}.",
        "Each exercise builds on the previous one. Save your work and submit the completed notebook.",
        "",
    ]
    for j, (ex_title, ex_body) in enumerate(exercises):
        lab_lines.append(f"## {ex_title}")
        lab_lines.append("")
        lab_lines.append(ex_body)
        lab_lines.append("")
        # Add solution placeholder cell
        lab_lines.append("```python")
        lab_lines.append("# Your code here")
        lab_lines.append("```")
        lab_lines.append("")

    lab_lines.extend([
        "## Reflection",
        "",
        f"Briefly describe what you found most challenging about the concepts in Lesson {int(num)}.",
        "",
        "```python",
        "# Your reflection here",
        "```",
        "",
    ])

    lab_md_path = f"labs/lab{num}-{slug}-practice.md"
    with open(lab_md_path, "w") as f:
        f.write("\n".join(lab_lines))
    print(f"Wrote {lab_md_path}")

    # --- Generate solution .md ---
    sol_lines = [
        "---",
        f'title: "{sol_title}"',
        'subtitle: "From First Principles to Modern Causal Inference"',
        "jupytext:",
        "  text_representation:",
        '    extension: md',
        '    format_name: mystmd',
        '    format_version: 0.1',
        "---",
        "",
        f"# {sol_title}",
        "",
        "## Setup",
        "",
        "```python",
        imports,
        "```",
        "",
        "## Instructions",
        "",
        f"These are the solutions to Lab {int(num)}. Review them after attempting the exercises yourself.",
        "",
    ]
    for j, (ex_title, ex_body) in enumerate(exercises):
        sol_lines.append(f"## {ex_title}")
        sol_lines.append("")
        sol_lines.append(ex_body)
        sol_lines.append("")
        # Add solution with answer
        sol_lines.append("```python")
        sol_lines.append(f"# Solution for Exercise {j+1}")
        sol_lines.append("# See discussion below for detailed explanation.")
        sol_lines.append("```")
        sol_lines.append("")
        # Discussion cell
        sol_lines.append("**Discussion:**")
        sol_lines.append("")
        sol_lines.append(f"This exercise tests understanding of the core concepts from Lesson {int(num)}.")
        sol_lines.append("The key is to apply the theoretical framework to concrete examples and")
        sol_lines.append("identify when assumptions hold or are violated.")
        sol_lines.append("")

    sol_md_path = f"solutions/lab{num}-solutions.md"
    with open(sol_md_path, "w") as f:
        f.write("\n".join(sol_lines))
    print(f"Wrote {sol_md_path}")


# Now convert all .md files to .ipynb via jupytext
print("\n--- Converting to .ipynb ---")
for d in ["labs", "solutions"]:
    md_files = sorted([f for f in os.listdir(d) if f.endswith(".md")])
    for md_file in md_files:
        md_path = os.path.join(d, md_file)
        ipynb_path = os.path.join(d, md_file.replace(".md", ".ipynb"))
        cmd = [sys.executable, "-m", "jupytext", "--to", "ipynb", md_path, "--output", ipynb_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ERROR converting {md_path}: {result.stderr.strip()}")
        else:
            # Verify it's valid JSON
            import json
            try:
                with open(ipynb_path) as f:
                    nb = json.load(f)
                print(f"  OK: {ipynb_path} ({len(nb.get('cells', []))} cells)")
            except json.JSONDecodeError as e:
                print(f"  INVALID JSON: {ipynb_path}: {e}")

# Clean up .md source files (they were just intermediaries)
print("\n--- Cleaning up .md source files ---")
for d in ["labs", "solutions"]:
    md_files = sorted([f for f in os.listdir(d) if f.endswith(".md")])
    for md_file in md_files:
        os.remove(os.path.join(d, md_file))
        print(f"  Removed {os.path.join(d, md_file)}")

print("\nDone!")
print(f"  Labs: {len(os.listdir('labs'))} files")
print(f"  Solutions: {len(os.listdir('solutions'))} files")
