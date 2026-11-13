import numpy as np
from .common import blur, gradient, laplace


def scale_space(im, sigmas):
    return np.array([s*s*laplace(blur(im,s)) for s in sigmas])


def extrema(response, threshold=.01):
    a=abs(response);valid=np.ones(a.shape,bool)
    valid[[0,-1]]=False;valid[:,[0,-1]]=False;valid[:,:,[0,-1]]=False
    c=a[1:-1,1:-1,1:-1]; local=np.ones(c.shape,bool)
    for ds in (-1,0,1):
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                if ds==dy==dx==0:continue
                local &= c>a[1+ds:a.shape[0]-1+ds,1+dy:a.shape[1]-1+dy,1+dx:a.shape[2]-1+dx]
    valid[1:-1,1:-1,1:-1] &= local
    return np.argwhere(valid & (a>threshold))


def derivatives3(a, p):
    p=np.array(p,dtype=int);v=a[tuple(p)];g=np.zeros(3);h=np.zeros((3,3))
    basis=np.eye(3,dtype=int)
    for i,e in enumerate(basis):
        g[i]=(a[tuple(p+e)]-a[tuple(p-e)])/2
        h[i,i]=a[tuple(p+e)]-2*v+a[tuple(p-e)]
        for j in range(i):
            f=basis[j]
            h[i,j]=h[j,i]=(a[tuple(p+e+f)]-a[tuple(p+e-f)]-a[tuple(p-e+f)]+a[tuple(p-e-f)])/4
    return g,h


def localize(response, candidates, max_iter=5):
    """Taylor fit of |sigma^2 LoG| in (log-scale index,y,x), relocating if needed."""
    a=abs(response);out=[];seen=set()
    for p0 in candidates:
        p=p0.copy()
        for _ in range(max_iter):
            if np.any(p<1) or np.any(p>=np.array(a.shape)-1):break
            g,h=derivatives3(a,p)
            if np.linalg.eigvalsh(h).max()>=-1e-10:break
            off=-np.linalg.solve(h,g)
            if np.max(abs(off))<=.5:
                key=tuple(p)
                if key not in seen:
                    out.append([*(p+off),a[tuple(p)]+.5*g@off]);seen.add(key)
                break
            p+=np.where(abs(off)>.5,np.sign(off),0).astype(int)
    return np.array(out,dtype=float).reshape(-1,4)


def harris(im, derivative_sigma=1., window_sigma=1.8, k=.04):
    gx,gy=gradient(blur(im,derivative_sigma))
    a,b,c=blur(gx*gx,window_sigma),blur(gx*gy,window_sigma),blur(gy*gy,window_sigma)
    det=a*c-b*b;trace=a+c
    return det,trace,det-k*trace*trace


def corners(response, fraction=.015, radius=3):
    h,w=response.shape;p=np.pad(response,radius,constant_values=-np.inf)
    maxima=response>max(0,float(response.max())*fraction)
    for dy in range(2*radius+1):
        for dx in range(2*radius+1):
            if dy==dx==radius:continue
            maxima &= response>=p[dy:dy+h,dx:dx+w]
    return np.argwhere(maxima)


def sift_filter(response, points, contrast=.025, edge_ratio=10):
    """SIFT spatial Hessian trace^2/det < (r+1)^2/r and contrast tests."""
    kept=[]
    for point in points:
        s,y,x,value=point;p=np.rint([s,y,x]).astype(int)
        _,h=derivatives3(response,p);spatial=h[1:,1:];det=np.linalg.det(spatial)
        ratio=np.trace(spatial)**2/det if det>0 else np.inf
        if value>=contrast and ratio<(edge_ratio+1)**2/edge_ratio:kept.append(point)
    return np.array(kept,dtype=float).reshape(-1,4)


def affine_image(im, angle=22, zoom=.82, gain=.85, offset=.07, noise=.008, seed=4):
    """Inverse-map a similarity transform with own bilinear interpolation."""
    h,w=im.shape[:2];theta=np.deg2rad(angle)
    a=zoom*np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    center=np.array([(w-1)/2,(h-1)/2]); y,x=np.mgrid[:h,:w]
    source=(np.stack([x,y],-1)-center)@np.linalg.inv(a).T+center
    sx,sy=source[...,0],source[...,1];x0=np.floor(sx).astype(int);y0=np.floor(sy).astype(int)
    valid=(sx>=0)&(sx<=w-1)&(sy>=0)&(sy<=h-1)
    xx=np.clip(x0,0,w-2);yy=np.clip(y0,0,h-2);fx=(sx-xx)[...,None];fy=(sy-yy)[...,None]
    out=(1-fy)*((1-fx)*im[yy,xx]+fx*im[yy,xx+1])+fy*((1-fx)*im[yy+1,xx]+fx*im[yy+1,xx+1])
    out=gain*out+offset+np.random.default_rng(seed).normal(0,noise,out.shape)
    out=np.clip(out,0,1);out[~valid]=.5
    return out,a,center,valid


def physical_points(points,sigmas):
    if not len(points):return np.empty((0,3))
    return np.column_stack([points[:,2],points[:,1],sigmas[0]*(sigmas[1]/sigmas[0])**points[:,0]])


def supported(points, sigmas, shape, valid=None):
    """Reject extrema whose 3-sigma smoothing support crosses an image/fill border."""
    h,w=shape[:2];keep=[]
    for point,(x,y,s) in zip(points,physical_points(points,sigmas)):
        r=int(np.ceil(3*s));xx=int(round(x));yy=int(round(y))
        if xx-r<0 or xx+r>=w or yy-r<0 or yy+r>=h:continue
        if valid is not None and not valid[yy-r:yy+r+1,xx-r:xx+r+1].all():continue
        keep.append(point)
    return np.array(keep,dtype=float).reshape(-1,4)


def repeatability(original,detected,a,center,zoom,valid):
    """Greedy one-to-one geometric matching; descriptors are not extracted."""
    expected=original.copy();expected[:,:2]=(original[:,:2]-center)@a.T+center;expected[:,2]*=zoom
    h,w=valid.shape;eligible=[]
    for i,(x,y,s) in enumerate(expected):
        margin=3*s
        if margin<x<w-margin and margin<y<h-margin:
            yy,xx=int(round(y)),int(round(x))
            # exclude artificial transform-fill boundaries from the visibility denominator
            r=int(np.ceil(margin))
            if valid[yy-r:yy+r+1,xx-r:xx+r+1].all():eligible.append(i)
    pairs=[]
    for i in eligible:
        if not len(detected):continue
        distances=np.linalg.norm(detected[:,:2]-expected[i,:2],axis=1)
        ratios=np.maximum(detected[:,2]/expected[i,2],expected[i,2]/detected[:,2])
        for j in np.flatnonzero((distances<=3)&(ratios<=np.sqrt(2))):pairs.append((distances[j],i,int(j)))
    used_i=set();used_j=set();errors=[]
    for d,i,j in sorted(pairs):
        if i not in used_i and j not in used_j:used_i.add(i);used_j.add(j);errors.append(float(d))
    return expected,{'eligible':len(eligible),'matches':len(errors),'repeatability':len(errors)/len(eligible) if eligible else 0.,'mean_error_px':float(np.mean(errors)) if errors else None,'position_tolerance_px':3,'scale_ratio_tolerance':float(np.sqrt(2))}
