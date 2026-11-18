import sys, os, numpy as np, time
from scipy.spatial import cKDTree
root = sys.argv[1]
def load(d):
    i = os.path.basename(d)
    tr = np.loadtxt(f"{d}/{i}.track", ndmin=2); nd = np.loadtxt(f"{d}/{i}.nodes", ndmin=2)
    ar = np.loadtxt(f"{d}/{i}.arcs", dtype=int, ndmin=2); ro = np.loadtxt(f"{d}/{i}.route", dtype=int, ndmin=1)
    return tr, nd, ar, ro
def xy(lon, lat, lon0, lat0):
    R = 6371000.0
    return np.radians(lon-lon0)*R*np.cos(np.radians(lat0)), np.radians(lat-lat0)*R
def seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx-ax, by-ay; L2 = dx*dx+dy*dy
    t = np.clip(((px-ax)*dx+(py-ay)*dy)/np.where(L2 == 0, 1, L2), 0, 1)
    return np.hypot(px-(ax+t*dx), py-(ay+t*dy))
agg = dict(pts=0, route_conn=0, route_pairs=0, nn_mono=0, miss={}, d=[], gaps=0, dup_route=0, rev=0)
trips = sorted(os.listdir(root))
for tid in trips:
    tr, nd, ar, ro = load(f"{root}/{tid}")
    lon0, lat0 = tr[:,0].mean(), tr[:,1].mean()
    px, py = xy(tr[:,0], tr[:,1], lon0, lat0); nx_, ny_ = xy(nd[:,0], nd[:,1], lon0, lat0)
    ax, ay, bx, by = nx_[ar[:,0]], ny_[ar[:,0]], nx_[ar[:,1]], ny_[ar[:,1]]
    # route connectivity
    agg['route_pairs'] += len(ro)-1; agg['route_conn'] += np.sum(ar[ro[:-1],1] == ar[ro[1:],0])
    agg['dup_route'] += len(ro) - len(set(ro.tolist()))
    agg['gaps'] += np.sum(np.diff(tr[:,2]) > 5)
    # monotonic labels
    rd = np.stack([seg_dist(px, py, ax[a], ay[a], bx[a], by[a]) for a in ro], 1)  # pts x route
    lab = np.empty(len(px), int); k = 0
    for i in range(len(px)):
        w = rd[i, k:k+15]; j = k+int(np.argmin(w)); lab[i] = j; k = j
    larc = ro[lab]; ld = rd[np.arange(len(px)), lab]; agg['d'].append(ld)
    glob = np.argmin(rd, 1); agg['nn_mono'] += np.sum(ro[glob] == larc)
    # candidates via densified kdtree
    L = np.hypot(bx-ax, by-ay); n = np.maximum(1, np.ceil(L/10).astype(int))
    idx = np.repeat(np.arange(len(ar)), n); f = np.concatenate([np.linspace(0, 1, m) for m in n])
    tree = cKDTree(np.c_[ax[idx]+f*(bx-ax)[idx], ay[idx]+f*(by-ay)[idx]])
    for R in (30, 50, 80):
        hits = tree.query_ball_point(np.c_[px, py], R+8)
        miss = 0; top1 = 0; nc = 0
        for i, h in enumerate(hits):
            c = np.unique(idx[h]);
            if len(c) == 0: miss += 1; continue
            dd = seg_dist(px[i], py[i], ax[c], ay[c], bx[c], by[c]); c = c[dd <= R]; dd = dd[dd <= R]
            nc += len(c)
            if larc[i] not in c: miss += 1
            elif c[np.argmin(dd)] == larc[i]: top1 += 1
            # reverse-twin of label present & at same dist?
        m = agg['miss'].setdefault(R, [0, 0, 0]); m[0] += miss; m[1] += top1; m[2] += nc
    # reverse twin rate: label arc has opposite arc
    pairs = set(map(tuple, ar.tolist())); agg['rev'] += sum((ar[a,1], ar[a,0]) in pairs for a in larc)
    agg['pts'] += len(px)
d = np.concatenate(agg['d']); P = agg['pts']
print(f"trips={len(trips)} pts={P}")
print(f"route consecutive arcs connected: {agg['route_conn']}/{agg['route_pairs']} = {agg['route_conn']/agg['route_pairs']:.3f}; repeated arcs in routes={agg['dup_route']}")
print(f"gaps>5s: {agg['gaps']}")
print("label dist pctl 50/90/95/99/max:", np.percentile(d, [50, 90, 95, 99, 100]).round(1), f" >50m: {np.mean(d>50):.3%}")
print(f"global-nearest-route-arc == monotonic label: {agg['nn_mono']/P:.3%}")
for R, (miss, top1, nc) in agg['miss'].items():
    print(f"R={R}m: miss={miss/P:.3%} nearest-arc Top1={top1/P:.3%} avg cands={nc/P:.1f}")
print(f"label arc has a reverse twin (2-way road): {agg['rev']/P:.3%}")
