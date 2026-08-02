#!/usr/bin/env python3
"""
Generate the foundational raster figures for Causal Inference 101.
Run this script to create all required figures.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

def create_core_figures():
    """Create the foundational figures used by the textbook."""

    # Figure 1: Ladder of Causation
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Draw ladder
    for i in range(3):
        y = 2 + i * 3
        ax.add_patch(mpatches.FancyBboxPatch((2, y), 6, 2,
                      boxstyle="round,pad=0.1",
                      facecolor=['lightyellow', 'lightblue', 'lightgreen'][i],
                      edgecolor='black', linewidth=2))

    # Add text
    ax.text(5, 3, 'Level 1: Association\n"What is Y given X?"\nP(Y|X)',
            ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(5, 6, 'Level 2: Intervention\n"What happens to Y if I do X?"\nP(Y|do(X))',
            ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(5, 9, 'Level 3: Counterfactuals\n"What would have happened if X were different?"\nP(Y_x|X=x\')',
            ha='center', va='center', fontsize=12, fontweight='bold')

    # Add arrows
    ax.annotate('', xy=(5, 5), xytext=(5, 4),
                arrowprops=dict(arrowstyle='->', lw=3, color='black'))
    ax.annotate('', xy=(5, 8), xytext=(5, 7),
                arrowprops=dict(arrowstyle='->', lw=3, color='black'))

    ax.set_title("Pearl's Ladder of Causation", fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('figures/ladder-of-causation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: figures/ladder-of-causation.png")

    # Figure 2: Confounding Example
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Nodes
    ax.add_patch(mpatches.Circle((5, 6), 0.8, facecolor='lightblue', edgecolor='black', linewidth=2))
    ax.add_patch(mpatches.Circle((2, 2), 0.8, facecolor='lightyellow', edgecolor='black', linewidth=2))
    ax.add_patch(mpatches.Circle((8, 2), 0.8, facecolor='lightyellow', edgecolor='black', linewidth=2))

    ax.text(5, 6, 'Z\n(Confounder)', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(2, 2, 'X\n(Treatment)', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(8, 2, 'Y\n(Outcome)', ha='center', va='center', fontsize=10, fontweight='bold')

    # Arrows
    ax.annotate('', xy=(2.7, 2.7), xytext=(4.3, 5.3),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    ax.annotate('', xy=(7.3, 2.7), xytext=(5.7, 5.3),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    ax.annotate('', xy=(7.2, 2), xytext=(2.8, 2),
                arrowprops=dict(arrowstyle='->', lw=2, color='gray', linestyle='dashed'))

    ax.set_title("Confounding: Z Causes Both X and Y", fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/01-confounding.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: figures/01-confounding.png")

    # Figure 3: Confounding Triangle
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Triangle
    triangle = plt.Polygon([[5, 8], [2, 3], [8, 3]],
                           facecolor='lightyellow', edgecolor='black', linewidth=2)
    ax.add_patch(triangle)

    # Nodes
    ax.add_patch(mpatches.Circle((5, 8), 0.6, facecolor='lightblue', edgecolor='black', linewidth=2))
    ax.add_patch(mpatches.Circle((2, 3), 0.6, facecolor='lightyellow', edgecolor='black', linewidth=2))
    ax.add_patch(mpatches.Circle((8, 3), 0.6, facecolor='lightyellow', edgecolor='black', linewidth=2))

    ax.text(5, 8, 'Z', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(2, 3, 'X', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(8, 3, 'Y', ha='center', va='center', fontsize=12, fontweight='bold')

    ax.set_title("The Confounding Triangle", fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/01-confounding-triangle.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: figures/01-confounding-triangle.png")

    # Figure 4: Pedagogical Order
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')

    steps = ['Problem', 'Intuition', 'History', 'Formal\nFramework',
             'Mathematics', 'Worked\nExamples', 'Python\nWorkshop',
             'Diagnostics', 'Interpretation', 'Practical\nApplication',
             'Limitations', 'Exercises', 'Projects']

    x_positions = np.linspace(1, 11, len(steps))
    y_positions = [4] * len(steps)

    for i, (x, y, step) in enumerate(zip(x_positions, y_positions, steps)):
        color = plt.cm.viridis(i / len(steps))
        ax.add_patch(mpatches.FancyBboxPatch((x-0.4, y-0.3), 0.8, 0.6,
                      boxstyle="round,pad=0.05", facecolor=color, edgecolor='black'))
        ax.text(x, y, step, ha='center', va='center', fontsize=8, fontweight='bold')

        if i < len(steps) - 1:
            ax.annotate('', xy=(x_positions[i+1]-0.5, y), xytext=(x+0.5, y),
                        arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))

    ax.set_title("Pedagogical Sequence Used in Every Lesson", fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/pedagogical-order.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: figures/pedagogical-order.png")

    print("\nAll foundational figures created successfully!")

if __name__ == "__main__":
    create_core_figures()
