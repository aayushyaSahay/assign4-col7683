"""Evaluation of generated results; references never enter segmentation costs."""
from pathlib import Path
import numpy as np
from PIL import Image
from .common import boundary


def read_seg(path):
    header,data=Path(path).read_text().split('data\n',1)
    fields=dict(line.split(maxsplit=1) for line in header.splitlines() if ' ' in line)
    a=np.full((int(fields['height']),int(fields['width'])),-1,dtype=np.int32)
    for line in data.splitlines():
        k,y,x0,x1=map(int,line.split());a[y,x0:x1+1]=k
    if (a<0).any():raise ValueError('incomplete segmentation')
    return a


def reference_masks(directory, shape):
    # Label 1 is the whole main insect in each of the five supplied color segmentations.
    return [np.asarray(Image.fromarray((read_seg(p)==1).astype('uint8')).resize(
        (shape[1],shape[0]),Image.Resampling.NEAREST),dtype=bool)
        for p in sorted(Path(directory).glob('*.seg'))]


def expand(mask, radius=2):
    h,w=mask.shape;p=np.pad(mask,radius);out=np.zeros_like(mask)
    for dy in range(-radius,radius+1):
        for dx in range(-radius,radius+1):
            if dx*dx+dy*dy<=radius*radius:
                out |= p[radius+dy:radius+dy+h,radius+dx:radius+dx+w]
    return out


def agreement(mask, references):
    rows=[];pred_boundary=boundary(mask);pred_expanded=expand(pred_boundary)
    for ref in references:
        intersection=np.count_nonzero(mask&ref);union=np.count_nonzero(mask|ref)
        rb=boundary(ref);precision=float(np.count_nonzero(pred_boundary&expand(rb))/max(1,pred_boundary.sum()))
        recall=float(np.count_nonzero(rb&pred_expanded)/max(1,rb.sum()))
        rows.append({'iou':intersection/max(1,union),'precision':intersection/max(1,int(mask.sum())),
                     'recall':intersection/max(1,int(ref.sum())),
                     'boundary_f1':2*precision*recall/max(1e-12,precision+recall)})
    return {**{k:float(np.mean([r[k] for r in rows])) for k in rows[0]},
            'iou_min':min(r['iou'] for r in rows),'iou_max':max(r['iou'] for r in rows),
            'foreground_pixels':int(mask.sum())}


def edge_agreement(edges, reference_boundaries):
    dilated=expand(edges);rows=[]
    for ref in reference_boundaries:
        p=np.count_nonzero(edges&expand(ref))/max(1,int(edges.sum()))
        r=np.count_nonzero(ref&dilated)/max(1,int(ref.sum()))
        rows.append((p,r,2*p*r/max(1e-12,p+r)))
    return dict(zip(['precision','recall','boundary_f1'],np.mean(rows,axis=0).tolist()))
