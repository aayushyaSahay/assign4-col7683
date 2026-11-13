"""Fixed manual inputs; shared by main experiments and parameter studies."""
import numpy as np

WING_SEEDS_XY = np.array([[303,135],[244,124],[170,121],[103,114],[72,126],
                          [66,145],[81,167],[119,182],[172,184],[232,169],[291,150]])
FG_RECTS = [(95,132,225,158),(150,160,183,171)]
BG_RECTS = [(15,15,450,70),(15,275,450,315),(380,225,470,265),(445,110,475,215)]
OBJECT_RECT = [35,80,450,270]


def source_rect(rect, shape):
    h,w=shape[:2]
    return tuple(np.rint(np.array(rect)*[w/481,h/321,w/481,h/321]).astype(int))


def rectangles_mask(rects, shape):
    mask=np.zeros(shape[:2],bool)
    for rect in rects:
        x0,y0,x1,y1=source_rect(rect,shape);mask[y0:y1,x0:x1]=True
    return mask


def region_seeds(shape):
    return rectangles_mask(FG_RECTS,shape),rectangles_mask(BG_RECTS,shape)


def wing_seeds(shape):
    h,w=shape[:2]
    return np.rint(WING_SEEDS_XY*[w/481,h/321]).astype(int)[:,::-1].tolist()
