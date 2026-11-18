"""Smoke test: the package and every subteam folder import."""

import importlib

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "topkd_imat",
        "topkd_imat.data",
        "topkd_imat.models",
        "topkd_imat.topology",
        "topkd_imat.training",
        "topkd_imat.analysis",
    ],
)
def test_module_imports(module):
    importlib.import_module(module)
