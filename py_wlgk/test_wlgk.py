import networkx as nx
import pytest
from wlgk import wlgk

SMALL_VALUE = 1e-5


# Sample Graphs for Testing
@pytest.fixture
def sample_graphs():
    G1 = nx.DiGraph()
    G1.add_edges_from([(1, 2), (2, 3), (3, 1)])

    G2 = nx.DiGraph()
    G2.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1)])

    G3 = nx.Graph()
    G3.add_edges_from([(1, 2), (2, 3), (3, 1)])

    G4 = nx.Graph()
    G4.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1)])

    return G1, G2, G3, G4


def test_wlgk_on_directed_graphs(sample_graphs):
    G1, G2, _, _ = sample_graphs
    similarity = wlgk(G1, G2)

    # Ensure it returns a value between 0 and 1 (inclusive)
    assert 0 - SMALL_VALUE <= similarity <= 1 + SMALL_VALUE


def test_wlgk_on_undirected_graphs(sample_graphs):
    _, _, G3, G4 = sample_graphs
    similarity = wlgk(G3, G4)

    # Ensure it returns a value between 0 and 1 (inclusive)
    assert 0 - SMALL_VALUE <= similarity <= 1 + SMALL_VALUE


def test_wlgk_on_same_graph(sample_graphs):
    G1, _, _, _ = sample_graphs
    similarity = wlgk(G1, G1)

    # The similarity between two identical graphs should be 1
    assert abs(1 - similarity) < SMALL_VALUE


def test_wlgk_on_empty_graphs():
    G1 = nx.DiGraph()
    G2 = nx.DiGraph()

    similarity = wlgk(G1, G2)

    # The similarity between two empty graphs should be 1
    assert abs(1 - similarity) < SMALL_VALUE
