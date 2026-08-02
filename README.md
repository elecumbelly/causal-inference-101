# Causal Inference 101

[Read the public book](https://causal-inference-101.vercel.app)

A practical, computation-supported introduction to causal reasoning, study
design, identification strategies, and modern estimation methods. The book
includes 24 lessons, curated videos, worked examples, exercises, datasets,
downloadable notebooks, and companion practice material.

## Preview the book locally

Use the pinned MyST dependency so the local preview matches Vercel exactly:

```console
npm install
npm run build
python3 -m http.server 3000 --directory _build/html
```

Open <http://localhost:3000>. Re-run `npm run build` after changing content or
layout files, then refresh the browser.

The build briefly starts internal MyST servers on other ports. Those ports are
expected and are not the website URL.

Jupyter Book 2 can also start an editing server with `jupyter-book start`, but
the static preview above is the release-equivalent path and includes the custom
desktop layout controls.

Do not add `.` to the command. In Jupyter Book 2, positional arguments are files
to export, so `jupyter-book build --html .` attempts to read the directory as a
file and fails with `EISDIR`.

If invoking Jupyter Book directly, do not add `.` to the build command. In
Jupyter Book 2, positional arguments are files to export, so
`jupyter-book build --html .` attempts to read the directory as a file and
fails with `EISDIR`. The valid direct command is:

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
