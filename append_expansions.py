#!/usr/bin/env python3
import os
import glob
import re
from pathlib import Path

EXPANSIONS_DIR = 'tmp_expansions'
LESSONS_DIR = 'lessons'

for exp_file in sorted(glob.glob(f"{EXPANSIONS_DIR}/*-expansion*.md")):
    basename = os.path.basename(exp_file)
    match = re.match(r'(\d+)-expansion', basename)
    if not match:
        continue
    lesson_num = int(match.group(1))

    lesson_files = glob.glob(f"{LESSONS_DIR}/{lesson_num:02d}-*.md")
    if not lesson_files:
        print(f"SKIP: No lesson file for lesson {lesson_num}")
        continue

    lesson_path = lesson_files[0]

    with open(exp_file, 'r') as f:
        expansion = f.read()

    with open(lesson_path, 'r') as f:
        lesson = f.read()

    if expansion in lesson:
        print(f"SKIP: {basename} is already present in lesson {lesson_num}")
        continue

    with open(lesson_path, 'a') as f:
        f.write(expansion)

    print(f"OK: Appended {basename} to lesson {lesson_num}")

print("\nFinal line counts:")
line_counts = []
for lesson_path in Path(LESSONS_DIR).glob('*.md'):
    line_count = len(lesson_path.read_text().splitlines())
    line_counts.append((line_count, lesson_path))
for line_count, lesson_path in sorted(line_counts):
    print(f"{line_count:5d} {lesson_path}")
