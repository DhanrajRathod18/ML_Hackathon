"""Directed road-network gap bridging and length-weighted route metrics."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

import networkx as nx
import numpy as np


EARTH_RADIUS_M = 6_371_008.8


def _node_array(nodes: Any) -> np.ndarray:
    try:
        node_values = np.asarray(nodes, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("nodes must be a two-column longitude/latitude array") from exc
    if node_values.ndim != 2 or node_values.shape[1] != 2:
        raise ValueError("nodes must be a two-column longitude/latitude array")
    if not np.all(np.isfinite(node_values)):
        raise ValueError("nodes contain missing or non-finite coordinates")
    if np.any(node_values[:, 0] < -180.0) or np.any(node_values[:, 0] > 180.0):
        raise ValueError("longitude coordinates must be within [-180, 180]")
    if np.any(node_values[:, 1] < -90.0) or np.any(node_values[:, 1] > 90.0):
        raise ValueError("latitude coordinates must be within [-90, 90]")
    return node_values


def _arc_array(arcs: Any) -> np.ndarray:
    try:
        arc_values = np.asarray(arcs)
    except (TypeError, ValueError) as exc:
        raise ValueError("arcs must contain integer from/to node pairs") from exc
    if arc_values.ndim != 2 or arc_values.shape[1] != 2:
        raise ValueError("arcs must contain integer from/to node pairs")
    if not np.issubdtype(arc_values.dtype, np.integer):
        raise ValueError("arcs must contain integer from/to node pairs")
    return arc_values.astype(np.int64)


def _xy_coordinates(nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project longitude/latitude with the same local equirectangular convention."""
    lon0 = float(np.mean(nodes[:, 0]))
    lat0 = float(np.mean(nodes[:, 1]))
    cos_lat0 = np.cos(np.radians(lat0))
    x = np.radians(nodes[:, 0] - lon0) * EARTH_RADIUS_M * cos_lat0
    y = np.radians(nodes[:, 1] - lat0) * EARTH_RADIUS_M
    return x, y


def _validate_network(arcs: Any, nodes: Any) -> tuple[np.ndarray, np.ndarray]:
    node_values = _node_array(nodes)
    arc_values = _arc_array(arcs)
    if len(arc_values) > 0:
        if np.any(arc_values[:, 0] < 0) or np.any(arc_values[:, 1] < 0):
            raise ValueError("arc node IDs must be non-negative")
        if np.any(arc_values[:, 0] >= len(node_values)):
            raise ValueError("arc contains an invalid from_node ID")
        if np.any(arc_values[:, 1] >= len(node_values)):
            raise ValueError("arc contains an invalid to_node ID")
    return arc_values, node_values


def build_arc_graph(arcs: Any, nodes: Any) -> nx.MultiDiGraph:
    """Build a directed arc-ID-keyed graph with physical edge lengths in metres."""
    arc_values, node_values = _validate_network(arcs, nodes)
    x, y = _xy_coordinates(node_values)

    graph = nx.MultiDiGraph()
    graph.add_nodes_from(range(len(node_values)))

    for arc_id, (from_node, to_node) in enumerate(arc_values):
        dx = x[to_node] - x[from_node]
        dy = y[to_node] - y[from_node]
        length_m = float(np.hypot(dx, dy))
        if not np.isfinite(length_m):
            raise ValueError("arc geometry is non-finite")
        graph.add_edge(
            int(from_node),
            int(to_node),
            key=int(arc_id),
            weight=max(0.0, length_m),
            length_m=max(0.0, length_m),
            arc_id=int(arc_id),
        )
    return graph


def _arc_lengths(arcs: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    arc_values, node_values = _validate_network(arcs, nodes)
    x, y = _xy_coordinates(node_values)
    lengths = np.empty(len(arc_values), dtype=float)
    for arc_id, (from_node, to_node) in enumerate(arc_values):
        dx = x[to_node] - x[from_node]
        dy = y[to_node] - y[from_node]
        length_m = float(np.hypot(dx, dy))
        if not np.isfinite(length_m):
            raise ValueError("arc geometry is non-finite")
        lengths[arc_id] = max(0.0, length_m)
    return lengths


def _normalize_arc_ids(pred_arcs: Any, n_arcs: int) -> list[int]:
    if pred_arcs is None:
        raise ValueError("predicted arc sequence cannot be None")
    try:
        if isinstance(pred_arcs, (str, bytes)):
            raise TypeError
        sequence = list(pred_arcs)
    except (TypeError, ValueError) as exc:
        raise ValueError("predicted arc sequence must be an iterable of arc IDs") from exc

    normalized: list[int] = []
    for arc_id in sequence:
        if isinstance(arc_id, (bool, np.bool_)) or not isinstance(arc_id, (int, np.integer)):
            raise ValueError("arc IDs must be integers")
        integer_id = int(arc_id)
        if integer_id < 0 or integer_id >= n_arcs:
            raise ValueError(f"invalid arc ID: {integer_id}")
        normalized.append(integer_id)
    return normalized


def bridge_route(pred_arcs: Sequence[int] | np.ndarray, arcs: Any, nodes: Any) -> list[int]:
    """Insert shortest directed connecting arcs between consecutive predictions."""
    arc_values, node_values = _validate_network(arcs, nodes)
    predicted = _normalize_arc_ids(pred_arcs, len(arc_values))
    if not predicted:
        return []

    graph = build_arc_graph(arc_values, node_values)
    bridged = [predicted[0]]
    path_cache: dict[tuple[int, int], list[int] | None] = {}

    for previous_id, next_id in zip(predicted, predicted[1:]):
        # Consecutive GPS points on the same arc are one traversal, not a loop back to its start.
        if previous_id == next_id:
            continue
        previous_end = int(arc_values[previous_id, 1])
        next_start = int(arc_values[next_id, 0])

        if previous_end == next_start:
            bridged.append(next_id)
            continue

        cache_key = (previous_end, next_start)
        if cache_key not in path_cache:
            try:
                path_cache[cache_key] = nx.shortest_path(
                    graph,
                    source=previous_end,
                    target=next_start,
                    weight="weight",
                )
            except (nx.NodeNotFound, nx.NetworkXNoPath):
                path_cache[cache_key] = None

        node_path = path_cache[cache_key]
        if node_path is None:
            bridged.append(next_id)
            continue

        for from_node, to_node in zip(node_path, node_path[1:]):
            edge_data = graph.get_edge_data(from_node, to_node)
            if not edge_data:
                continue
            chosen = min(
                edge_data.values(),
                key=lambda data: (float(data["weight"]), int(data["arc_id"])),
            )
            arc_id = int(chosen["arc_id"])
            bridged.append(arc_id)

        bridged.append(next_id)

    return bridged


def route_precision_recall(
    pred_arcs: Sequence[int] | np.ndarray,
    reference_arcs: Sequence[int] | np.ndarray,
    arcs: Any,
    nodes: Any,
) -> tuple[float, float]:
    """Calculate length-weighted arc precision and recall using physical lengths."""
    arc_values, node_values = _validate_network(arcs, nodes)
    predicted = _normalize_arc_ids(pred_arcs, len(arc_values))
    reference = _normalize_arc_ids(reference_arcs, len(arc_values))

    if not predicted or not reference:
        return 0.0, 0.0

    lengths = _arc_lengths(arc_values, node_values)
    predicted_counts = Counter(predicted)
    reference_counts = Counter(reference)

    shared_length = sum(
        min(predicted_counts[arc_id], reference_counts[arc_id]) * float(lengths[arc_id])
        for arc_id in predicted_counts.keys() & reference_counts.keys()
    )
    predicted_length = sum(count * float(lengths[arc_id]) for arc_id, count in predicted_counts.items())
    reference_length = sum(count * float(lengths[arc_id]) for arc_id, count in reference_counts.items())

    precision = shared_length / predicted_length if predicted_length > 0.0 else 0.0
    recall = shared_length / reference_length if reference_length > 0.0 else 0.0
    return float(precision), float(recall)
