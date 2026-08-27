"""
Shared fixtures for the QMD math-pipeline test suite.

Generation is deterministic but not free (n<=4 takes tens of seconds), so the
GenerationResults are session-scoped: every test module shares one run.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qmd.core import run_generation, to_matrix  # noqa: E402


@pytest.fixture(scope="session")
def r3():
    return run_generation(max_vertices=3, bound=2)


@pytest.fixture(scope="session")
def r4():
    return run_generation(max_vertices=4, bound=2)


# --- well-known matrices ---------------------------------------------------

@pytest.fixture(scope="session")
def A2():
    return to_matrix([[0, 1], [-1, 0]])


@pytest.fixture(scope="session")
def A3():
    return to_matrix([[0, 1, 0], [-1, 0, 1], [0, -1, 0]])


@pytest.fixture(scope="session")
def D4():
    return to_matrix([[0, 1, 1, 1], [-1, 0, 0, 0], [-1, 0, 0, 0], [-1, 0, 0, 0]])


@pytest.fixture(scope="session")
def kronecker():
    return to_matrix([[0, 2], [-2, 0]])


@pytest.fixture(scope="session")
def markov():
    return to_matrix([[0, 2, -2], [-2, 0, 2], [2, -2, 0]])
