import sys, os, numpy as np
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"probe3.py")).read().split("P = 0; t0")[0])
rows=[]
for tid in sorted(os.listdir(root)):
    tr, nd, ar, ro = load(f"{root}/{tid}")
    lon0, lat0 = tr[:,0].mean(), tr[:,1].mean()
    px, py = xy(tr[:,0], tr[:,1], lon0, lat0); nx_, ny_ = xy(nd[:,0], nd[:,1], lon0, lat0)
    ax, ay, bx, by = nx_[ar[:,0]], ny_[ar[:,0]], nx_[ar[:,1]], ny_[ar[:,1]]
    rd = np.stack([seg(px, py, ax[a], ay[a], bx[a], by[a]) for a in ro], 1)
    lab = dp_align(rd, 8); ld = rd[np.arange(len(px)), lab]
    mind = rd.min(1)  # distance to ANY route arc
    far = ld > 50
    # where are far points: start/end/middle?
    fi = np.where(far)[0]
    conn = np.mean(ar[ro[:-1],1]==ar[ro[1:],0])
    rows.append((tid, len(px), len(ro), far.mean(), (mind>50).mean(), fi.min()/len(px) if len(fi) else -1, fi.max()/len(px) if len(fi) else -1, conn, np.max(np.diff(tr[:,2]))))
rows.sort(key=lambda r:-r[3])
print("trip pts route far%  anyRouteArc>50m%  firstFar lastFar routeConn maxGap_s")
for r in rows[:15]: print(f"{r[0]} {r[1]:5d} {r[2]:4d} {r[3]:6.1%} {r[4]:6.1%} {r[5]:5.2f} {r[6]:5.2f} {r[7]:.3f} {r[8]:.0f}")
print("trips with far>5%:", sum(r[3]>0.05 for r in rows), "of", len(rows))
