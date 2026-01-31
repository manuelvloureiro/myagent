"""Fixtures for import-swappable compatibility tests.

These tests verify API compatibility between myagent and langgraph.
Every test must pass against both implementations.

Run with myagent (default):
    uv run pytest tests/test_compat/

Run with real langgraph:
    uv run pytest tests/test_compat/ --langgraph

Run both back-to-back:
    uv run pytest tests/test_compat/ -v && uv run pytest tests/test_compat/ --langgraph -v
"""

import importlib

import pytest


def pytest_addoption(parser):
    parser.addoption("--langgraph", action="store_true", default=False, help="Run against real langgraph")


@pytest.fixture
def use_langgraph(request):
    return request.config.getoption("--langgraph")


@pytest.fixture
def graph_module(use_langgraph):
    """Returns the graph module (myagent.graph or langgraph.graph)."""
    if use_langgraph:
        return importlib.import_module("langgraph.graph")
    return importlib.import_module("myagent.graph")


@pytest.fixture
def StateGraph(graph_module):
    return graph_module.StateGraph


@pytest.fixture
def START(graph_module):
    return graph_module.START


@pytest.fixture
def END(graph_module):
    return graph_module.END


@pytest.fixture
def MessagesState(graph_module):
    return graph_module.MessagesState


@pytest.fixture
def add_messages(graph_module):
    return graph_module.add_messages


@pytest.fixture
def InMemorySaver(use_langgraph):
    if use_langgraph:
        return importlib.import_module("langgraph.checkpoint.memory").MemorySaver
    return importlib.import_module("myagent.checkpoint.memory").InMemorySaver


@pytest.fixture
def GraphRecursionError(use_langgraph):
    if use_langgraph:
        return importlib.import_module("langgraph.errors").GraphRecursionError
    return importlib.import_module("myagent.errors").GraphRecursionError


@pytest.fixture
def RetryPolicy(use_langgraph):
    if use_langgraph:
        return importlib.import_module("langgraph.types").RetryPolicy
    return importlib.import_module("myagent.types").RetryPolicy
