"""Geometry helpers.

Projection choice: local equirectangular projection centred on the trip centroid.
Trips are a few km to a few tens of km across, so the distortion versus a true
geodesic is far below GPS noise (<0.1 % over ~20 km), while it is vectorised and
needs no CRS lookup for trips that lie in different UTM zones / countries.
"""
import numpy as np

EARTH_R = 6371008.8  # mean Earth radius in metres


def to_xy(lon, lat, lon0, lat0):
    x = np.radians(np.asarray(lon) - lon0) * EARTH_R * np.cos(np.radians(lat0))
    y = np.radians(np.asarray(lat) - lat0) * EARTH_R
    return x, y


def point_segment(px, py, ax, ay, bx, by):
    """Distance from point(s) P to segment(s) AB and projection fraction t in [0,1]. Broadcasts."""
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = ((px - ax) * dx + (py - ay) * dy) / np.where(L2 == 0, 1.0, L2)
    t = np.clip(t, 0.0, 1.0)
    return np.hypot(px - (ax + t * dx), py - (ay + t * dy)), t


def ang_diff(a, b):
    """Absolute smallest difference between two angles (radians) -> [0, pi]."""
    return np.abs((a - b + np.pi) % (2 * np.pi) - np.pi)


class TripGeometry:
    """Metric coordinates of one trip's GPS points and its road network."""

    def __init__(self, trip):
        tr = trip["track"]
        self.lon0, self.lat0 = float(tr.lon.mean()), float(tr.lat.mean())
        self.px, self.py = to_xy(tr.lon.values, tr.lat.values, self.lon0, self.lat0)
        self.t = tr.t.values.astype(float)
        nx, ny = to_xy(trip["nodes"][:, 0], trip["nodes"][:, 1], self.lon0, self.lat0)
        a = trip["arcs"]
        self.arcs = a
        self.ax, self.ay = nx[a[:, 0]], ny[a[:, 0]]
        self.bx, self.by = nx[a[:, 1]], ny[a[:, 1]]
        self.arc_len = np.hypot(self.bx - self.ax, self.by - self.ay)
        self.arc_bearing = np.arctan2(self.by - self.ay, self.bx - self.ax)
        n_nodes = len(trip["nodes"])
        self.out_deg = np.bincount(a[:, 0], minlength=n_nodes)[a[:, 1]]  # choices after this arc
        self.in_deg = np.bincount(a[:, 1], minlength=n_nodes)[a[:, 0]]
        # reverse twin: arc (v,u) exists for arc (u,v)  -> two-way road
        key = a[:, 0] * n_nodes + a[:, 1]
        rkey = a[:, 1] * n_nodes + a[:, 0]
        order = np.argsort(key)
        pos = np.searchsorted(key[order], rkey)
        pos = np.clip(pos, 0, len(key) - 1)
        found = key[order][pos] == rkey
        self.twin = np.where(found, order[pos], -1)
