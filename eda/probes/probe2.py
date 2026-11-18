import sys, os, numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
root = sys.argv[1]
def load(d):
    i = os.path.basename(d)
    return (np.loadtxt(f"{d}/{i}.track", ndmin=2), np.loadtxt(f"{d}/{i}.nodes", ndmin=2),
            np.loadtxt(f"{d}/{i}.arcs", dtype=int, ndmin=2), np.loadtxt(f"{d}/{i}.route", dtype=int, ndmin=1))
def xy(lon, lat, lon0, lat0):
    R = 6371000.0; return np.radians(lon-lon0)*R*np.cos(np.radians(lat0)), np.radians(lat-lat0)*R
def seg(px, py, ax, ay, bx, by):
    dx, dy = bx-ax, by-ay; L2 = dx*dx+dy*dy
    t = np.clip(((px-ax)*dx+(py-ay)*dy)/np.where(L2 == 0, 1, L2), 0, 1)
    return np.hypot(px-(ax+t*dx), py-(ay+t*dy))
res = {w: [0, 0] for w in (5, 15, 40, 10**9)}; P = 0; und = 0; head = 0; far_runs = []; n_ok = 0
for tid in sorted(os.listdir(root)):
    tr, nd, ar, ro = load(f"{root}/{tid}")
    lon0, lat0 = tr[:,0].mean(), tr[:,1].mean()
    px, py = xy(tr[:,0], tr[:,1], lon0, lat0); nx_, ny_ = xy(nd[:,0], nd[:,1], lon0, lat0)
    ax, ay, bx, by = nx_[ar[:,0]], ny_[ar[:,0]], nx_[ar[:,1]], ny_[ar[:,1]]
    rd = np.stack([seg(px, py, ax[a], ay[a], bx[a], by[a]) for a in ro], 1)
    labs = {}
    for w in res:
        lab = np.empty(len(px), int); k = 0
        for i in range(len(px)):
            j = k+int(np.argmin(rd[i, k:k+w])); lab[i] = j; k = j
        dd = rd[np.arange(len(px)), lab]; res[w][0] += np.sum(dd > 50); res[w][1] += np.sum(dd)
        labs[w] = lab
    larc = ro[labs[15]]; ld = rd[np.arange(len(px)), labs[15]]
    # far-run lengths
    f = ld > 50; 
    if f.any():
        e = np.diff(np.r_[0, f.astype(int), 0]); s = np.where(e == 1)[0]; t = np.where(e == -1)[0]; far_runs += list(t-s)
    # heading of GPS (central difference over +-2)
    hx = np.r_[px[2:], [px[-1]]*2] - np.r_[[px[0]]*2, px[:-2]]; hy = np.r_[py[2:], [py[-1]]*2] - np.r_[[py[0]]*2, py[:-2]]
    gh = np.arctan2(hy, hx); moving = np.hypot(hx, hy) > 2
    ah = np.arctan2(by-ay, bx-ax)
    L = np.hypot(bx-ax, by-ay); n = np.maximum(1, np.ceil(L/10).astype(int))
    idx = np.repeat(np.arange(len(ar)), n); fr = np.concatenate([np.linspace(0, 1, m) for m in n])
    tree = cKDTree(np.c_[ax[idx]+fr*(bx-ax)[idx], ay[idx]+fr*(by-ay)[idx]])
    hits = tree.query_ball_point(np.c_[px, py], 58)
    for i, h in enumerate(hits):
        c = np.unique(idx[h])
        if len(c) == 0: continue
        dd = seg(px[i], py[i], ax[c], ay[c], bx[c], by[c]); keep = dd <= 50; c, dd = c[keep], dd[keep]
        if len(c) == 0 or larc[i] not in c: continue
        n_ok += 1
        best = c[np.argmin(dd)]
        if best == larc[i] or (ar[best,0] == ar[larc[i],1] and ar[best,1] == ar[larc[i],0]): und += 1
        hd = np.abs((ah[c]-gh[i]+np.pi) % (2*np.pi) - np.pi)
        score = dd + (20*(hd > np.pi/2) if moving[i] else 0)   # crude: penalise wrong-direction by 20 m
        if c[np.argmin(score)] == larc[i]: head += 1
    P += len(px)
print(f"pts={P}")
for w, (nf, sd) in res.items(): print(f"label window {w}: >50m={nf/P:.3%} mean label dist={sd/P:.2f} m")
print(f"far(>50m) runs: count={len(far_runs)} median len={np.median(far_runs) if far_runs else 0} max={max(far_runs) if far_runs else 0}")
print(f"of points whose label is in R=50 set ({n_ok/P:.2%} of all):")
print(f"  nearest arc ignoring direction correct: {und/n_ok:.3%}")
print(f"  nearest + crude heading rule correct:  {head/n_ok:.3%}")
