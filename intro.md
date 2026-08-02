# Causal Inference 101
## From First Principles to Modern Causal Inference

```{figure} assets/causal-network-hero.jpg
:name: causal-network-hero
:alt: Observational data resolving into a directional causal network after an intervention
:width: 100%

Finding the causal signal inside a world of associations.
```

*By a team of causal inference researchers and educators*

---

**A Practical Guide to Understanding, Implementing, and Evaluating Causal Methods**

---

## Welcome

You are about to embark on a journey through one of the most important intellectual frameworks of the last century: **causal inference**.

This is a practical, computation-supported textbook designed to help you move from observing patterns to reasoning carefully about *why* they exist—and what may happen under an intervention.

By the time you finish this course, you will be able to:

- Ask precise causal questions
- Formulate causal estimands
- Understand counterfactual reasoning
- Recognise threats to causal identification
- Choose appropriate causal methods
- Implement those methods in Python
- Interpret estimates correctly
- Evaluate assumptions critically
- Perform diagnostics that reveal problems
- Communicate conclusions honestly
- Judge whether causal claims are believable

---

## Why This Book Exists

Every day, decisions are made based on data. Companies launch products, governments enact policies, doctors prescribe treatments, and algorithms allocate resources. The question that matters most is rarely "what happened?" but rather **"what would happen if we did X instead?"**

That is a causal question.

The tools for answering causal questions rigorously span statistics, epidemiology, economics and computer science. This book provides a guided route through that toolkit, taught from first principles with downloadable notebooks and practical examples.

```{figure} figures/ladder-of-causation.png
:name: ladder-of-causation
:alt: Pearl's Ladder of Causation
:width: 400px

Pearl's Ladder of Causation: Association, Intervention, and Counterfactuals.
```

---

## Who This Book Is For

This book is designed for:

- **Data scientists** who want to move beyond correlation to causation
- **Graduate students** in economics, statistics, public health, political science, or sociology
- **Industry practitioners** who need to evaluate causal claims from experiments or observational studies
- **Researchers** who want a comprehensive reference for causal methods
- **Anyone curious** about why things happen and how we can know

### Prerequisites

- Basic statistics (mean, variance, regression)
- Python programming (we teach the causal methods, not Python basics)
- Linear algebra (matrix multiplication, basic matrix algebra)
- Calculus (derivatives, integrals at an intuitive level)

No prior knowledge of causal inference is assumed. We build everything from first principles.

---

## Course Philosophy

This is a practical **Causal Inference 101** course.

Every method we teach answers six questions:

1. **What problem does it solve?**
2. **Why do simpler approaches fail?**
3. **What assumptions are required?**
4. **How is it implemented?**
5. **What exactly does it estimate?**
6. **When does it fail, and how can those failures be detected?**

The guiding questions throughout this book are:

> *Which causal method should I use?*
> *Why does it work?*
> *What assumptions make it valid?*
> *Should I believe the resulting causal claim?*

---

## Pedagogical Approach

We teach in a consistent order throughout:

```{figure} figures/pedagogical-order.png
:name: pedagogical-order
:alt: Pedagogical order from problem to project
:width: 500px

The pedagogical sequence used in every lesson.
```

| Stage | Description |
|-------|-------------|
| **Problem** | Start with a real-world puzzle |
| **Intuition** | Build understanding before formalism |
| **History** | Why was this method invented? |
| **Formal Framework** | Definitions, notation, assumptions |
| **Mathematics** | Rigorous treatment |
| **Worked Examples** | Medicine, economics, technology, policy |
| **Python Workshop** | Complete executable notebooks |
| **Diagnostics** | How to check if the method is working |
| **Interpretation** | What the results actually mean |
| **Practical Application** | Real-world consulting scenarios |
| **Limitations** | When the method fails |
| **Exercises** | Test your understanding |
| **Projects** | Apply what you've learned |

**We never introduce notation before motivation. We never introduce equations before intuition. We never introduce code before mathematics.**

---

## Book Structure

The book is organized into six parts:

### Part I: Thinking Causally
Foundations of causal reasoning. You will learn why causality matters, how to think about potential outcomes, how to draw causal diagrams, and what confounding really means.

### Part II: Identification Strategies
The core toolkit for causal inference. You will master randomized experiments, understand selection bias, learn regression for causal adjustment, and become proficient with propensity score methods.

### Part III: Natural Experiments
When randomization is impossible, nature sometimes provides it. You will learn instrumental variables, difference-in-differences, regression discontinuity, and synthetic control methods.

### Part IV: Advanced Causal Methods
More complex causal questions require more sophisticated tools. You will learn mediation analysis, time-varying treatments, missing data methods, and sensitivity analysis.

### Part V: Modern Causal Machine Learning
The frontier of causal inference. You will learn double machine learning, heterogeneous treatment effects, causal forests, and Bayesian causal inference.

### Part VI: Practical Causal Inference
How to actually do causal inference in the real world. You will learn study design, how to read research papers, industry applications across healthcare/economics/marketing/technology/policy/AI, and complete a capstone project.

---

## Interactive Features

This book is designed to be interactive:

- **Executable notebooks**: Run the code yourself
- **Downloadable notebooks**: Take them with you
- **Collapsible solutions**: Check your work
- **Interactive quizzes**: Test your knowledge
- **Glossary links**: Understand every term
- **Search**: Find anything instantly
- **Dark mode**: Read comfortably

---

## How to Use This Book

### For Self-Study
Work through the lessons in order. Complete all exercises. Do the labs. The capstone project at the end will test everything you've learned.

### For a Course
Instructors can use this as a complete course textbook. Each lesson is designed for approximately 2-3 hours of instruction, with additional time for labs and projects.

### For Reference
Use the glossary and table of contents to find specific topics. Each lesson is self-contained enough to be read independently, though concepts build progressively.

---

## The Datasets

Throughout this book, we use the same core datasets so you can revisit them from different methodological perspectives:

| Dataset | Used In | Description |
|---------|---------|-------------|
| **Framingham Heart Study** | Lesson 1 | Cardiovascular risk factors |
| **Lalonde NSW** | Lessons 2, 8 | Job training experiment |
| **Oregon Medicaid Lottery** | Lesson 5 | Health insurance natural experiment |
| **Kidney Stone Treatment** | Lesson 4 | Simpson's paradox classic |
| **Draft Lottery** | Lesson 9 | Vietnam War draft instruments |
| **Card & Krueger Minimum Wage** | Lesson 10 | Minimum wage effects |
| **California Tobacco Control** | Lesson 12 | Tobacco tax effects |

We also create synthetic datasets throughout to illustrate specific concepts.

---

## A Note on Honesty

Causal inference is fundamentally about being honest with ourselves and our audience about what we know and what we don't know. Every causal claim rests on assumptions. Some assumptions are reasonable, others are not. Part of learning causal inference is learning to:

1. **State your assumptions explicitly**
2. **Evaluate whether they are plausible**
3. **Conduct sensitivity analyses**
4. **Communicate uncertainty honestly**
5. **Resist the temptation to overclaim**

We will model this behavior throughout the book. When we make assumptions, we say so. When we are uncertain, we say so. When results might be interpreted multiple ways, we present all interpretations.

---

## Let's Begin

Turn to [Lesson 1: Why Causality Matters](lessons/01-why-causality-matters.md) to start your journey into causal inference.

The question that drives everything that follows is deceptively simple:

> *What would have happened if...?*

That question—that *counterfactual*—is the heart of causal inference. Everything else is machinery for answering it honestly.

Welcome to Causal Inference 101.
