from collections import deque
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import maximum_flow


def fit_gaussian(x, regularization=4e-4):
    if len(x)<2: raise ValueError('at least two training pixels required')
    mean = x.mean(0); centered = x-mean
    cov = centered.T@centered/len(x) + regularization*np.eye(x.shape[-1])
    return mean, cov


def nll(x, model):
    mean,cov = model; d=x-mean
    return .5*(np.einsum('...i,ij,...j->...',d,np.linalg.inv(cov),d)+np.linalg.slogdet(cov)[1]+len(mean)*np.log(2*np.pi))


def likelihood_classify(im, fg, bg, regularization=4e-4):
    models = [fit_gaussian(im[bg],regularization),fit_gaussian(im[fg],regularization)]
    costs = np.stack([nll(im,m) for m in models],axis=-1)
    return costs.argmin(-1).astype(bool),costs,models


def kmeans(im, fg, bg, max_iter=100, constrain=True):
    means=np.array([im[bg].mean(0),im[fg].mean(0)])
    history=[]; wc=[]; previous=None
    for _ in range(max_iter):
        d=((im[:,:,None,:]-means)**2).sum(-1)
        labels=d.argmin(-1).astype(bool)
        if constrain: labels[fg]=True; labels[bg]=False
        for k in (0,1):
            if np.any(labels==k): means[k]=im[labels==k].mean(0)
        wc.append(float(((im-means[labels.astype(int)])**2).sum()))
        history.append(labels.copy())
        if previous is not None and np.array_equal(labels,previous): break
        previous=labels.copy()
    return history,wc,means


def pairwise(im, lam, sigma):
    if sigma<=0 or lam<0: raise ValueError('sigma must be positive, lambda nonnegative')
    h,w=im.shape[:2]; ids=np.arange(h*w).reshape(h,w)
    ps=[];qs=[];vs=[]
    for dy,dx in [(0,1),(1,0),(1,1),(1,-1)]:
        ys=slice(0,h-dy); xs=slice(max(0,-dx),min(w,w-dx))
        yt=slice(dy,h); xt=slice(max(0,dx),min(w,w+dx))
        delta=((im[ys,xs]-im[yt,xt])**2).sum(-1)
        ps.append(ids[ys,xs].ravel());qs.append(ids[yt,xt].ravel())
        vs.append((lam*np.exp(-delta/(2*sigma*sigma))/np.hypot(dy,dx)).ravel())
    return np.concatenate(ps),np.concatenate(qs),np.concatenate(vs)


def graphcut(im, costs, fg, bg, lam=2, sigma=.08, scale=256):
    """Binary Potts min cut. Source-side pixels are foreground (class 1)."""
    if np.any(fg & bg): raise ValueError('inconsistent seed masks')
    h,w=im.shape[:2];n=h*w;ids=np.arange(n)
    # Subtract a label-independent constant so all terminal capacities are nonnegative.
    unary=costs-costs.min(-1,keepdims=True)
    unary=np.rint(unary.reshape(n,2)*scale).astype(np.int64)
    p,q,v=pairwise(im,lam,sigma); v=np.rint(v*scale).astype(np.int64)
    # Flipping a wrongly labelled seed saves more than all its incident finite costs.
    incident=np.zeros(n,dtype=np.int64)
    np.add.at(incident,p,v);np.add.at(incident,q,v)
    hard=unary.max(1)+incident+1
    if hard.max()>=2**31-1: raise ValueError('capacity exceeds int32; reduce scale')
    unary[fg.ravel(),0]=hard[fg.ravel()];unary[bg.ravel(),1]=hard[bg.ravel()]
    rows=np.concatenate([np.full(n,n),ids,p,q]);cols=np.concatenate([ids,np.full(n,n+1),q,p])
    values=np.concatenate([unary[:,0],unary[:,1],v,v])
    graph=coo_matrix((values,(rows,cols)),shape=(n+2,n+2),dtype=np.int64).tocsr()
    result=maximum_flow(graph,n,n+1,method='dinic')
    residual=(graph-result.flow).tocsr();seen=np.zeros(n+2,bool);seen[n]=True;queue=deque([n])
    while queue:
        u=queue.popleft()
        for j in range(residual.indptr[u],residual.indptr[u+1]):
            vtx=residual.indices[j]
            if residual.data[j]>0 and not seen[vtx]:seen[vtx]=True;queue.append(vtx)
    labels=seen[:n].reshape(h,w)
    return labels,{'integer_cut':int(result.flow_value),'capacity_scale':scale,'max_hard_capacity':int(hard.max())}


def energy(im,costs,labels,lam,sigma):
    shifted=costs-costs.min(-1,keepdims=True)
    unary=np.take_along_axis(shifted,labels.astype(int)[...,None],-1).sum()
    p,q,v=pairwise(im,lam,sigma);flat=labels.ravel()
    return float(unary+v[flat[p]!=flat[q]].sum())


def rectangle_em(im, rect, max_iter=60, regularization=.008, ring_width=4):
    """Rectangle-only Gaussian/ML alternation requested by Q2(d), no graph cuts."""
    x0,y0,x1,y1=rect; inside=np.zeros(im.shape[:2],bool);inside[y0:y1,x0:x1]=True
    inner=np.zeros_like(inside);inner[y0+ring_width:y1-ring_width,x0+ring_width:x1-ring_width]=True
    ring=inside & ~inner
    models=[fit_gaussian(im[ring],regularization),fit_gaussian(im[inner],regularization)]
    hist=[]; stats=[];prev=None
    for i in range(max_iter):
        costs=np.stack([nll(im,m) for m in models],-1)
        labels=(costs[:,:,1]<costs[:,:,0]) & inside
        if labels.sum()<2 or (~labels).sum()<2: raise ValueError('collapsed Gaussian class')
        hist.append(labels.copy())
        stats.append({'iteration':i+1,'foreground_pixels':int(labels.sum()),'changed':int(np.count_nonzero(labels != prev)) if prev is not None else int(labels.size)})
        if prev is not None and np.array_equal(labels,prev):break
        models=[fit_gaussian(im[~labels],regularization),fit_gaussian(im[labels],regularization)];prev=labels.copy()
    return hist,stats,inner,ring
