import heapq
import numpy as np
from .common import blur, gradient, laplace

NEIGHBORS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]


def marr_hildreth(im, sigma=1.4, threshold=.01):
    response = laplace(blur(im, sigma))
    p = np.pad(response, 1, mode='reflect'); h, w = im.shape
    strength = np.zeros_like(im)
    for dy, dx in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        a = p[1+dy:1+dy+h, 1+dx:1+dx+w]
        b = p[1-dy:1-dy+h, 1-dx:1-dx+w]
        strength = np.maximum(strength, np.where(a*b <= 0, abs(a-b), 0))
    return response, (strength > threshold), strength


def link_costs(im, weights=(.43, .43, .14), sigma=1.4, threshold=.01, gamma=1):
    _, edges, _ = marr_hildreth(im, sigma, threshold)
    gx, gy = gradient(blur(im, sigma)); mag = np.hypot(gx, gy)
    g = np.clip(mag/(mag.max()+1e-12), 0, 1)
    tx, ty = -gy/(mag+1e-12), gx/(mag+1e-12)
    h, w = im.shape; costs = np.full((8, h, w), np.inf)
    for k, (dy, dx) in enumerate(NEIGHBORS):
        ys = slice(max(0, -dy), min(h, h-dy)); xs = slice(max(0, -dx), min(w, w-dx))
        yt = slice(max(0, dy), min(h, h+dy)); xt = slice(max(0, dx), min(w, w+dx))
        d = np.hypot(dy, dx)
        a = np.arccos(np.clip(abs((dx*tx[ys,xs]+dy*ty[ys,xs])/d), 0, 1))
        b = np.arccos(np.clip(abs((dx*tx[yt,xt]+dy*ty[yt,xt])/d), 0, 1))
        z = 1 - .5*(edges[ys,xs].astype(float)+edges[yt,xt])
        gm = 1 - .5*(g[ys,xs]**gamma+g[yt,xt]**gamma)
        ori = (a+b)/np.pi
        costs[k,ys,xs] = d*(weights[0]*z+weights[1]*gm+weights[2]*ori+1e-5)
    return costs


def dijkstra(costs, seed, target=None):
    _, h, w = costs.shape
    sy, sx = seed
    if not (0 <= sy < h and 0 <= sx < w): raise ValueError('seed outside image')
    dist = np.full((h,w), np.inf); parent = np.full((h,w), -1, dtype=np.int32)
    dist[sy,sx] = 0; heap = [(0., sy, sx)]
    while heap:
        d, y, x = heapq.heappop(heap)
        if d != dist[y,x]: continue
        if target == (y,x): break
        for k, (dy,dx) in enumerate(NEIGHBORS):
            yy, xx = y+dy, x+dx
            if not (0 <= yy < h and 0 <= xx < w): continue
            nd = d+costs[k,y,x]
            if nd < dist[yy,xx]:
                dist[yy,xx] = nd; parent[yy,xx] = y*w+x
                heapq.heappush(heap, (nd, yy, xx))
    return dist, parent


def trace(parent, seed, target):
    w = parent.shape[1]; p = target; path = [p]
    while p != seed:
        n = int(parent[p])
        if n < 0: raise ValueError('unreachable target')
        p = divmod(n, w); path.append(p)
        if len(path) > parent.size: raise ValueError('cycle in predecessor tree')
    return np.array(path[::-1])


def closed_path(costs, seeds):
    paths = []; total = 0.
    for s,t in zip(seeds, seeds[1:]+seeds[:1]):
        dist, parent = dijkstra(costs, tuple(s), tuple(t))
        paths.append(trace(parent, tuple(s), tuple(t))[:-1]); total += dist[tuple(t)]
    path = np.concatenate(paths); path = np.vstack([path, path[0]])
    return path, total
