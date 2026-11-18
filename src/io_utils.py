"""Raw-file loading for PathSnap trips.

Each trip folder <id>/ holds <id>.track (lon lat t), <id>.nodes (lon lat),
<id>.arcs (from_node to_node) and, for labelled partitions, <id>.route (arc ids).
Every trip ships its OWN road network, so node/arc ids are only valid inside the trip.
"""
import os
import numpy as np
import pandas as pd


def list_trips(partition_dir):
    return sorted(d for d in os.listdir(partition_dir)
                  if os.path.isdir(os.path.join(partition_dir, d)))


def load_trip(trip_dir, with_route=True):
    tid = os.path.basename(os.path.normpath(trip_dir))
    p = lambda ext: os.path.join(trip_dir, f"{tid}.{ext}")
    track = pd.DataFrame(np.loadtxt(p("track"), ndmin=2), columns=["lon", "lat", "t"])
    nodes = np.loadtxt(p("nodes"), ndmin=2)
    arcs = np.loadtxt(p("arcs"), dtype=np.int64, ndmin=2)
    route = None
    if with_route and os.path.exists(p("route")):
        route = np.loadtxt(p("route"), dtype=np.int64, ndmin=1)
    return {"trip_id": tid, "track": track, "nodes": nodes, "arcs": arcs, "route": route}
