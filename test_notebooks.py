#!/usr/bin/env python3
"""Execute the dependency-light foundational notebook smoke suite."""

import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FOUNDATIONS = [
    '01-why-causality-matters.ipynb',
    '02-potential-outcomes.ipynb',
    '03-directed-acyclic-graphs.ipynb',
    '04-confounding.ipynb',
    '05-randomized-controlled-trials.ipynb',
]


def main():
    with tempfile.TemporaryDirectory(prefix='causal-notebooks-') as temporary:
        test_root = Path(temporary)
        notebook_dir = test_root / 'notebooks'
        notebook_dir.mkdir()
        (test_root / 'figures').mkdir()

        for filename in FOUNDATIONS:
            source = ROOT / 'notebooks' / filename
            target = notebook_dir / filename
            shutil.copy2(source, target)
            print(f'EXECUTE: {filename}', flush=True)
            subprocess.run(
                ['jupyter', 'execute', str(target), '--inplace', '--timeout=60'],
                check=True,
            )

    print(f'PASS: Executed {len(FOUNDATIONS)} foundational notebooks')


if __name__ == '__main__':
    main()
