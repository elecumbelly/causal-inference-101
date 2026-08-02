#!/usr/bin/env python3
"""Smoke-test a deployed Causal Inference 101 site."""

import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen


ROUTES = [
    '',
    'why-causality-matters',
    'potential-outcomes',
    'directed-acyclic-graphs',
    'confounding',
    'randomized-controlled-trials',
    'selection-bias-collider-bias',
    'regression-causal-adjustment',
    'propensity-scores',
    'instrumental-variables',
    'difference-in-differences',
    'regression-discontinuity',
    'synthetic-control',
    'mediation-analysis',
    'heterogeneous-effects',
    'sensitivity-analysis',
    'longitudinal-time-varying',
    'structural-causal-models',
    'causal-discovery',
    'causal-inference-ml',
    'bayesian-causal-inference',
    'external-validity',
    'decision-theory',
    'fairness',
    'capstone-project',
    'glossary',
]


def fetch(base_url, route):
    url = urljoin(f'{base_url.rstrip("/")}/', route)
    request = Request(url, headers={'User-Agent': 'causal-inference-release-test/1.0'})
    with urlopen(request, timeout=30) as response:
        return response.status, response.geturl(), response.read()


def main():
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python3 test_deployment.py <base-url>')

    base_url = sys.argv[1]
    failures = []
    homepage = b''

    for route in ROUTES:
        try:
            status, final_url, body = fetch(base_url, route)
            if status != 200:
                failures.append(f'/{route}: HTTP {status}')
            if route == '':
                homepage = body
            print(f'PASS {status}: {final_url}')
        except (HTTPError, URLError, TimeoutError) as error:
            failures.append(f'/{route}: {error}')

    for route in ['sitemap.xml', 'robots.txt', 'favicon.ico']:
        try:
            status, final_url, _ = fetch(base_url, route)
            if status != 200:
                failures.append(f'/{route}: HTTP {status}')
            print(f'PASS {status}: {final_url}')
        except (HTTPError, URLError, TimeoutError) as error:
            failures.append(f'/{route}: {error}')

    if b'Causal Inference 101' not in homepage:
        failures.append('homepage title is missing')
    target_host = urlparse(base_url).hostname
    is_local_check = target_host in {'localhost', '127.0.0.1'}
    if not is_local_check and (
        b'localhost:3000' in homepage or b'localhost:3100' in homepage
    ):
        failures.append('homepage leaks a local development URL')

    if failures:
        for failure in failures:
            print(f'FAIL: {failure}')
        raise SystemExit(1)

    print(f'PASS: verified {len(ROUTES) + 3} public endpoints')


if __name__ == '__main__':
    main()
