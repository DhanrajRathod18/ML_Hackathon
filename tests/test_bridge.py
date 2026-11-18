import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bridge import build_arc_graph, bridge_route, route_precision_recall


def assert_directionally_connected(arcs: np.ndarray, route: list[int]) -> None:
    for previous_id, next_id in zip(route, route[1:]):
        assert arcs[previous_id, 1] == arcs[next_id, 0]


def test_empty_predictions():
    nodes = np.array([[77.0, 12.0], [77.1, 12.0]])
    arcs = np.array([[0, 1]])

    assert bridge_route([], arcs, nodes) == []
    assert route_precision_recall([], [], arcs, nodes) == (0.0, 0.0)
    assert route_precision_recall([], [0], arcs, nodes) == (0.0, 0.0)
    assert route_precision_recall([0], [], arcs, nodes) == (0.0, 0.0)


def test_single_predicted_arc():
    nodes = np.array([[77.0, 12.0], [77.1, 12.0]])
    arcs = np.array([[0, 1]])

    assert bridge_route([0], arcs, nodes) == [0]
    assert route_precision_recall([0], [0], arcs, nodes) == (1.0, 1.0)


def test_already_connected_arcs():
    nodes = np.array([[77.0, 12.0], [77.1, 12.0], [77.2, 12.0]])
    arcs = np.array([[0, 1], [1, 2]])
    route = bridge_route([0, 1], arcs, nodes)

    assert route == [0, 1]
    assert_directionally_connected(arcs, route)


def test_disconnected_arcs_with_valid_connecting_path():
    nodes = np.array([[77.0, 12.0], [77.1, 12.0], [77.2, 12.0], [77.3, 12.0]])
    arcs = np.array([[0, 1], [1, 2], [2, 3], [0, 3]])
    route = bridge_route([0, 2], arcs, nodes)

    assert route == [0, 1, 2]
    assert_directionally_connected(arcs, route)


def test_no_directed_path():
    nodes = np.array([[77.0, 12.0], [77.1, 12.0], [77.2, 12.0]])
    arcs = np.array([[0, 1], [2, 1]])
    route = bridge_route([0, 1], arcs, nodes)

    assert route == [0, 1]


def test_reverse_direction_only_connection_is_rejected():
    nodes = np.array([[77.0, 12.0], [77.1, 12.0], [77.2, 12.0]])
    arcs = np.array([[0, 1], [2, 1]])
    route = bridge_route([0, 1], arcs, nodes)

    assert route == [0, 1]
    assert arcs[1, 0] != arcs[0, 1]


def test_invalid_prediction_and_reference_arc_ids():
    nodes = np.array([[77.0, 12.0], [77.1, 12.0]])
    arcs = np.array([[0, 1]])

    for invalid in (-1, 1, True, 1.0, "0", None):
        with pytest.raises(ValueError):
            bridge_route([invalid], arcs, nodes)
        with pytest.raises(ValueError):
            route_precision_recall([0], [invalid], arcs, nodes)
    with pytest.raises(ValueError):
        route_precision_recall([0.0], [0], arcs, nodes)


def test_repeated_arcs_are_preserved_in_order():
    nodes = np.array([[77.0, 12.0], [77.1, 12.0]])
    arcs = np.array([[0, 1], [1, 0]])

    route = bridge_route([0, 1, 0], arcs, nodes)
    assert route == [0, 1, 0]
    assert_directionally_connected(arcs, route)

    precision, recall = route_precision_recall([0, 0], [0, 1, 0], arcs, nodes)
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(2.0 / 3.0)


@pytest.mark.parametrize(
    ("predicted", "expected"),
    [
        ([10, 10, 10], [10]),
        ([10, 10, 15], [10, 15]),
        ([10, 15, 10], [10, 15, 14, 10]),
    ],
)
def test_consecutive_duplicate_predictions_collapse_without_losing_revisits(
    predicted, expected
):
    nodes = np.array([[77.0, 12.0], [77.1, 12.0], [77.2, 12.0]])
    arcs = np.zeros((16, 2), dtype=int)
    arcs[10] = [0, 1]
    arcs[14] = [2, 0]
    arcs[15] = [1, 2]

    route = bridge_route(predicted, arcs, nodes)

    assert route == expected
    assert_directionally_connected(arcs, route)


def test_valid_connecting_path_preserves_repeated_original_arc_id():
    nodes = np.array([[77.0, 12.0], [77.1, 12.0], [77.2, 12.0]])
    arcs = np.array([[0, 1], [1, 2], [2, 0]])

    route = bridge_route([0, 2, 1, 2], arcs, nodes)

    assert route == [0, 1, 2, 0, 1, 2]
    assert route.count(1) == 2
    assert route.count(2) == 2
    assert route.count(0) == 2
    assert_directionally_connected(arcs, route)


def test_parallel_arcs_force_equal_weight_bridge_selection():
    nodes = np.array([[77.0, 12.0], [77.1, 12.0], [77.2, 12.0], [77.3, 12.0]])
    arcs = np.array([[0, 1], [1, 2], [1, 2], [2, 3]])

    graph = build_arc_graph(arcs, nodes)
    assert graph.number_of_edges() == 4
    assert graph.has_edge(1, 2, 1)
    assert graph.has_edge(1, 2, 2)

    route = bridge_route([0, 3], arcs, nodes)
    assert route[0] == 0
    assert route[-1] == 3
    assert route[1] in (1, 2)
    assert_directionally_connected(arcs, route)
    assert route[1] in graph.get_edge_data(1, 2)


def test_parallel_arc_ties_are_deterministic():
    nodes = np.array([[77.0, 12.0], [77.1, 12.0], [77.2, 12.0]])
    arcs = np.array([[0, 1], [1, 2], [1, 2]])

    first = bridge_route([0, 2], arcs, nodes)
    second = bridge_route([0, 2], arcs, nodes)
    assert first == second
    assert first[1] in (1, 2)


def test_coordinate_ranges():
    arcs = np.array([[0, 1]])
    valid = np.array([[77.0, 12.0], [78.0, 13.0]])
    assert build_arc_graph(arcs, valid).number_of_edges() == 1

    for invalid_nodes in (
        np.array([[181.0, 12.0], [77.0, 12.0]]),
        np.array([[-181.0, 12.0], [77.0, 12.0]]),
        np.array([[77.0, 91.0], [77.0, 12.0]]),
        np.array([[77.0, -91.0], [77.0, 12.0]]),
        np.array([[77.0, np.nan], [77.0, 12.0]]),
    ):
        with pytest.raises(ValueError):
            build_arc_graph(arcs, invalid_nodes)


def test_zero_length_and_malformed_geometry():
    zero_nodes = np.array([[77.0, 12.0], [77.0, 12.0]])
    zero_arcs = np.array([[0, 1]])
    assert build_arc_graph(zero_arcs, zero_nodes).edges[0, 1, 0]["weight"] == 0.0
    assert route_precision_recall([0], [0], zero_arcs, zero_nodes) == (0.0, 0.0)

    with pytest.raises(ValueError):
        build_arc_graph(np.array([[0, 1]]), np.array([[77.0, 12.0]]))
    with pytest.raises(ValueError):
        build_arc_graph(np.array([[0, np.nan]]), np.array([[77.0, 12.0], [77.1, 12.0]]))
    with pytest.raises(ValueError):
        build_arc_graph(np.array([[0, 2]]), np.array([[77.0, 12.0], [77.1, 12.0]]))
    with pytest.raises(ValueError):
        build_arc_graph(np.array([[-1, 1]]), np.array([[77.0, 12.0], [77.1, 12.0]]))


def test_route_metrics_empty_nonoverlap_and_shared_bounds():
    nodes = np.array([[77.0, 12.0], [77.1, 12.0], [77.2, 12.0]])
    arcs = np.array([[0, 1], [1, 2]])

    assert route_precision_recall([0], [1], arcs, nodes) == (0.0, 0.0)
    precision, recall = route_precision_recall([0, 1], [0], arcs, nodes)
    assert precision == pytest.approx(0.5)
    assert recall == pytest.approx(1.0)
    precision, recall = route_precision_recall([0], [0, 1], arcs, nodes)
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(0.5)

    precision, recall = route_precision_recall([0, 1, 0], [0, 1], arcs, nodes)
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
