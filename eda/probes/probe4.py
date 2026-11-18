import sys, os, numpy as np, time
from scipy.spatial import cKDTree
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
def dp_align(rd, S):
    N, M = rd.shape; INF = 1e18
    cost = np.full(M, INF); cost[:S] = rd[0, :S]  # start near beginning of route
    back = np.zeros((N, M), np.int32)
    for i in range(1, N):
        best = cost.copy(); arg = np.arange(M)
        for s in range(1, S+1):
            sh = np.full(M, INF); sh[s:] = cost[:-s]
            m = sh < best; best[m] = sh[m]; arg[m] = (np.arange(M)-s)[m]
        cost = best + rd[i]; back[i] = arg
    j = int(np.argmin(cost[max(0, M-S):]) + max(0, M-S))  # end near route end
    lab = np.empty(N, int)
    for i in range(N-1, -1, -1): lab[i] = j; j = back[i, j]
    return lab
P = 0; t0 = time.time(); stats = {}
for S in (8,):
    far = 0; tot = 0; und = 0; head = 0; nearest_dir = 0; n_ok = 0; miss = 0; P = 0
    for tid in sorted(os.listdir(root)):
        tr, nd, ar, ro = load(f"{root}/{tid}")
        lon0, lat0 = tr[:,0].mean(), tr[:,1].mean()
        px, py = xy(tr[:,0], tr[:,1], lon0, lat0); nx_, ny_ = xy(nd[:,0], nd[:,1], lon0, lat0)
        ax, ay, bx, by = nx_[ar[:,0]], ny_[ar[:,0]], nx_[ar[:,1]], ny_[ar[:,1]]
        rd = np.stack([seg(px, py, ax[a], ay[a], bx[a], by[a]) for a in ro], 1)
        lab = dp_align(rd, S); larc = ro[lab]; ld = rd[np.arange(len(px)), lab]
        far += np.sum(ld > 50); tot += ld.sum()
        hx = np.r_[px[2:], [px[-1]]*2] - np.r_[[px[0]]*2, px[:-2]]; hy = np.r_[py[2:], [py[-1]]*2] - np.r_[[py[0]]*2, py[:-2]]
        gh = np.arctan2(hy, hx); moving = np.hypot(hx, hy) > 2; ah = np.arctan2(by-ay, bx-ax)
        L = np.hypot(bx-ax, by-ay); n = np.maximum(1, np.ceil(L/10).astype(int))
        idx = np.repeat(np.arange(len(ar)), n); fr = np.concatenate([np.linspace(0, 1, m) for m in n])
        tree = cKDTree(np.c_[ax[idx]+fr*(bx-ax)[idx], ay[idx]+fr*(by-ay)[idx]])
        for i, h in enumerate(tree.query_ball_point(np.c_[px, py], 58)):
            c = np.unique(idx[h])
            if len(c): 
                dd = seg(px[i], py[i], ax[c], ay[c], bx[c], by[c]); k = dd <= 50; c, dd = c[k], dd[k]
            if len(c) == 0 or larc[i] not in c: miss += 1; continue
            n_ok += 1; best = c[np.argmin(dd)]
            if best == larc[i]: nearest_dir += 1
            if best == larc[i] or (ar[best,0] == ar[larc[i],1] and ar[best,1] == ar[larc[i],0]): und += 1
            hd = np.abs((ah[c]-gh[i]+np.pi) % (2*np.pi) - np.pi)
            score = dd + (30*(hd > np.pi/2) if moving[i] else 0)
            if c[np.argmin(score)] == larc[i]: head += 1
            spd=np.hypot(hx[i],hy[i])/4; b=min(int(spd//3),6); hd_l=np.abs((ah[larc[i]]-gh[i]+np.pi)%(2*np.pi)-np.pi)
            stats.setdefault(b,[0,0]); stats[b][0]+=1; stats[b][1]+= hd_l<np.pi/2
        P += len(px)
    print(f"DP S={S}: label>50m={far/P:.3%} mean label dist={tot/P:.2f}m | miss@50={miss/P:.3%} | nearest(directed)={nearest_dir/n_ok:.2%} nearest(undirected)={und/n_ok:.2%} nearest+heading={head/n_ok:.2%}  [{time.time()-t0:.0f}s]")

for b in sorted(stats): print(f"speed {b*3}-{b*3+3} m/s: n={stats[b][0]} label-arc direction agrees with GPS heading: {stats[b][1]/stats[b][0]:.2%}")
