# Causal Inference 101

[Read the public book](https://causal-inference-101.vercel.app)

A practical, computation-supported introduction to causal reasoning, study
design, identification strategies, and modern estimation methods. The book
includes 24 lessons, curated videos, worked examples, exercises, datasets,
downloadable notebooks, and companion practice material.

## Preview the book locally

Install Jupyter Book 2, change into this directory, and run:

```console
jupyter-book start
```

Open the URL printed after `Server started`, normally <http://localhost:3000>.
Jupyter Book also starts an internal content server, normally on port 3100; that
second port is expected and is not the website URL.

Do not add `.` to the command. In Jupyter Book 2, positional arguments are files
to export, so `jupyter-book build --html .` attempts to read the directory as a
file and fails with `EISDIR`.

To produce static HTML in `_build/html`, run:

```console
jupyter-book build --html
```

## Project checks

```console
python3 test_all.py
```

Run the dependency-light notebook execution smoke suite separately because
Jupyter opens local kernel ports:

```console
python3 test_notebooks.py
```

Advanced notebooks additionally require the specialist libraries imported in
their first code cell (for example `linearmodels`, `econml`, `pymc`, or a causal
discovery package). The structural test compiles every notebook; the execution
suite covers the five foundational notebooks supported by the base environment.

## Production deployment

The public site is built with the pinned MyST CLI dependency in `package.json`.
This keeps local, continuous-deployment, and Vercel builds on the same version:

```console
npm install
npm run build
```

Vercel uses `vercel.json` to publish the generated `_build/html` directory.
After deployment, verify every lesson and public metadata endpoint with:

```console
python3 test_deployment.py https://your-deployment.example
```
