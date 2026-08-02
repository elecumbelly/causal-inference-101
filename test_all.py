#!/usr/bin/env python3
"""Tests for Causal Inference 101 project."""

import os
import glob
import sys
from pathlib import Path
import re
import json
import yaml
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent

def test_book_config():
    """Test that the book has deterministic navigation and theme config."""
    myst_path = PROJECT_DIR / 'myst.yml'
    with open(myst_path, 'r') as f:
        config = yaml.safe_load(f)

    toc = config.get('project', {}).get('toc', [])
    toc_files = []
    for entry in toc:
        if 'file' in entry:
            toc_files.append(entry['file'])
        toc_files.extend(child['file'] for child in entry.get('children', []))

    expected_lessons = {
        f'lessons/{number:02d}-{slug}.md'
        for number, slug in [
            (1, 'why-causality-matters'),
            (2, 'potential-outcomes'),
            (3, 'directed-acyclic-graphs'),
            (4, 'confounding'),
            (5, 'randomized-controlled-trials'),
            (6, 'selection-bias-collider-bias'),
            (7, 'regression-causal-adjustment'),
            (8, 'propensity-scores'),
            (9, 'instrumental-variables'),
            (10, 'difference-in-differences'),
            (11, 'regression-discontinuity'),
            (12, 'synthetic-control'),
            (13, 'mediation-analysis'),
            (14, 'heterogeneous-effects'),
            (15, 'sensitivity-analysis'),
            (16, 'longitudinal-time-varying'),
            (17, 'structural-causal-models'),
            (18, 'causal-discovery'),
            (19, 'causal-inference-ml'),
            (20, 'bayesian-causal-inference'),
            (21, 'external-validity'),
            (22, 'decision-theory'),
            (23, 'fairness'),
            (24, 'capstone-project'),
        ]
    }
    if not toc or toc_files[0] != 'intro.md':
        print("FAIL: intro.md is not the first table-of-contents entry")
        return False
    if not expected_lessons.issubset(toc_files):
        print("FAIL: myst.yml table of contents does not include all lessons")
        return False
    if config.get('site', {}).get('template') != 'book-theme':
        print("FAIL: myst.yml does not select the book theme")
        return False
    if 'causal-inference-101.vercel.app' not in config.get('site', {}).get('domains', []):
        print("FAIL: myst.yml does not declare the production domain")
        return False

    print("PASS: Book theme and navigation are configured")
    return True


def test_production_build_config():
    """Guard the static export against MyST's temporary localhost origin."""
    package = json.loads((PROJECT_DIR / 'package.json').read_text())
    build_command = package.get('scripts', {}).get('build', '')
    origin_script = PROJECT_DIR / 'scripts' / 'set-production-origin.mjs'

    if 'set-production-origin.mjs' not in build_command:
        print("FAIL: npm build does not normalize the production origin")
        return False
    if not origin_script.is_file():
        print("FAIL: production-origin normalization script is missing")
        return False

    print("PASS: Production build normalizes MyST's temporary origin")
    return True

def test_lesson_expansion():
    """Test that generated expansion blocks have not been appended twice."""
    lessons_dir = PROJECT_DIR / 'lessons'
    failed = []
    for expansion_file in sorted((PROJECT_DIR / 'tmp_expansions').glob('*-expansion*.md')):
        lesson_number = int(expansion_file.name.split('-', 1)[0])
        lesson_file = next(lessons_dir.glob(f'{lesson_number:02d}-*.md'))
        occurrences = lesson_file.read_text().count(expansion_file.read_text())
        if occurrences != 1:
            failed.append((lesson_file.name, expansion_file.name, occurrences))

    if failed:
        for lesson, expansion, occurrences in failed:
            print(f"FAIL: {lesson} contains {occurrences} copies of {expansion}")
        return False

    print("PASS: Every generated expansion appears exactly once")
    return True

def test_datasets():
    """Test that all datasets exist and have data."""
    datasets_dir = os.path.join(PROJECT_DIR, 'datasets')
    csv_files = sorted(glob.glob(os.path.join(datasets_dir, '*.csv')))

    if len(csv_files) < 24:
        print(f"FAIL: Only {len(csv_files)} datasets found (need 24)")
        return False

    failed = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if len(df) < 10:
                failed.append((os.path.basename(csv_file), len(df)))
        except Exception as e:
            failed.append((os.path.basename(csv_file), str(e)))

    if failed:
        for name, issue in failed:
            print(f"FAIL: {name} - {issue}")
        return False

    print(f"PASS: All {len(csv_files)} datasets exist with valid data")
    return True

def test_content_quality():
    """Test that all lessons have required pedagogical sections."""
    lessons_dir = os.path.join(PROJECT_DIR, 'lessons')
    lesson_files = sorted(glob.glob(os.path.join(lessons_dir, '*.md')))

    required_sections = ['Opening Story', 'Learning Objectives', 'Worked Example', 'Exercises', 'Watch and Connect']

    failed = []
    for index, lesson_file in enumerate(lesson_files, start=1):
        with open(lesson_file, 'r') as f:
            content = f.read()

        missing = []
        for section in required_sections:
            if section not in content:
                missing.append(section)
        practice_section = 'Python Workshop' if index < 8 else 'Companion Practice'
        if practice_section not in content:
            missing.append(practice_section)
        if re.search(r'\bTODO\b|\bPlaceholder\b|^\s*pass\s*$', content, re.MULTILINE):
            missing.append('finished code (placeholder found)')

        if missing:
            failed.append((os.path.basename(lesson_file), missing))

    if failed:
        for name, missing in failed:
            print(f"FAIL: {name} missing: {', '.join(missing)}")
        return False

    print(f"PASS: All {len(lesson_files)} lessons have required sections")
    return True

def test_media_and_design():
    """Test that each lesson has curated video media and design assets exist."""
    failed = []
    for lesson_file in sorted((PROJECT_DIR / 'lessons').glob('*.md')):
        content = lesson_file.read_text()
        youtube_links = re.findall(r'https://www\.youtube\.com/(?:watch\?v=[\w-]{11}|playlist\?list=[\w-]+)[^ )]*', content)
        if len(youtube_links) != 1:
            failed.append(f'{lesson_file.name}: expected one YouTube recommendation, found {len(youtube_links)}')

    for asset in ['assets/custom.css', 'assets/causal-network-hero.jpg', 'assets/logo.png']:
        if not (PROJECT_DIR / asset).is_file():
            failed.append(f'missing design asset: {asset}')

    stylesheet = (PROJECT_DIR / 'assets' / 'custom.css').read_text()
    design_guards = {
        'editorial design tokens': '--ci-display:',
        'dark theme': 'html.dark body',
        'mobile layout': '@media (max-width: 767px)',
        'compact mobile outline': 'max-height: 18rem !important',
        'reduced-motion support': '@media (prefers-reduced-motion: no-preference)',
    }
    for feature, marker in design_guards.items():
        if marker not in stylesheet:
            failed.append(f'missing design feature: {feature}')

    if failed:
        for issue in failed:
            print(f"FAIL: {issue}")
        return False

    print("PASS: All lessons have curated video media and design assets")
    return True

def test_methodology_and_figures():
    """Guard corrected claims and ensure instructional figures are wired in."""
    corpus = {path.name: path.read_text() for path in (PROJECT_DIR / 'lessons').glob('*.md')}
    failed = []
    guards = {
        '08-propensity-scores.md': ['does **not** balance omitted variables'],
        '10-difference-in-differences.md': ['Modern DiD with Staggered Adoption'],
        '11-regression-discontinuity.md': ['Scholarship Eligibility', 'robust bias-corrected'],
        '13-mediation-analysis.md': ['Natural Direct Effect', 'Natural Indirect Effect'],
        '17-structural-causal-models.md': [
            "getattr(nx, 'is_d_separator'",
            'Exchange actions and observations',
            'intercepts_all_directed_paths',
        ],
        '20-bayesian-causal-inference.md': ['do not turn an unidentified causal effect'],
        '24-capstone-project.md': ['The Six-Stage Causal Workflow'],
    }
    for filename, required in guards.items():
        for phrase in required:
            if phrase not in corpus[filename]:
                failed.append(f'{filename}: missing corrected concept "{phrase}"')

    if 'nx.d_separated' in corpus['17-structural-causal-models.md']:
        failed.append('17-structural-causal-models.md: uses removed NetworkX d_separated API')

    scm_block = re.search(
        r'### Identification Algorithm\s+```python\n(.*?)```',
        corpus['17-structural-causal-models.md'],
        re.DOTALL,
    )
    if not scm_block:
        failed.append('17-structural-causal-models.md: identification example not found')
    else:
        try:
            namespace = {}
            exec(scm_block.group(1), namespace)
            dag = namespace['CausalDAG']()
            dag.graph.add_edges_from([
                ('U', 'T'), ('U', 'Y'), ('T', 'M'), ('M', 'Y')
            ])
            if not dag.front_door_criterion('T', 'Y', 'M'):
                failed.append('17-structural-causal-models.md: rejects a valid front-door graph')
            dag.add_edge('T', 'Y')
            if dag.front_door_criterion('T', 'Y', 'M'):
                failed.append('17-structural-causal-models.md: accepts a direct path around mediator')
        except Exception as error:
            failed.append(f'17-structural-causal-models.md: example failed ({error})')

    referenced = set()
    for filename, content in corpus.items():
        for figure in re.findall(r'\.\./figures/instructional/([\w-]+\.svg)', content):
            referenced.add(figure)
            if not (PROJECT_DIR / 'figures' / 'instructional' / figure).is_file():
                failed.append(f'{filename}: missing figure {figure}')
    if len(referenced) < 12:
        failed.append(f'expected at least 12 instructional figures, found {len(referenced)}')

    if failed:
        for issue in failed:
            print(f"FAIL: {issue}")
        return False
    print(f"PASS: Methodology guards and {len(referenced)} instructional figures")
    return True

def test_notebook_source():
    """Validate and compile every notebook code cell without executing kernels."""
    failed = []
    notebooks = sorted((PROJECT_DIR / 'notebooks').glob('*.ipynb'))
    for notebook in notebooks:
        try:
            data = json.loads(notebook.read_text())
        except (json.JSONDecodeError, OSError) as error:
            failed.append(f'{notebook.name}: invalid notebook JSON ({error})')
            continue
        for index, cell in enumerate(data.get('cells', []), start=1):
            if cell.get('cell_type') != 'code':
                continue
            source = ''.join(cell.get('source', []))
            if re.search(r'\bTODO\b|\bPlaceholder\b|^\s*pass\s*$', source, re.MULTILINE):
                failed.append(f'{notebook.name} cell {index}: placeholder code')
            try:
                compile(source, f'{notebook.name}:cell-{index}', 'exec')
            except SyntaxError as error:
                failed.append(f'{notebook.name} cell {index}: {error.msg}')
    if failed:
        for issue in failed:
            print(f"FAIL: {issue}")
        return False
    print(f"PASS: All code cells compile in {len(notebooks)} notebooks")
    return True

if __name__ == '__main__':
    os.chdir(PROJECT_DIR)

    print("Running tests...")
    print("=" * 50)

    results = []
    results.append(("Book Config", test_book_config()))
    results.append(("Production Build Config", test_production_build_config()))
    results.append(("Lesson Expansion", test_lesson_expansion()))
    results.append(("Datasets", test_datasets()))
    results.append(("Content Quality", test_content_quality()))
    results.append(("Media and Design", test_media_and_design()))
    results.append(("Methodology and Figures", test_methodology_and_figures()))
    results.append(("Notebook Source", test_notebook_source()))

    print("=" * 50)
    passed = sum(1 for _, result in results if result)
    total = len(results)

    if passed == total:
        print(f"ALL {total} TESTS PASSED")
    else:
        print(f"{passed}/{total} tests passed")
        for name, result in results:
            if not result:
                print(f"  FAILED: {name}")
        sys.exit(1)
