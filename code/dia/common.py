from pathlib import Path
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load(path, size=320):
    im = Image.open(path).convert('RGB')
    im.thumbnail((size, size), Image.Resampling.LANCZOS)
    return np.asarray(im, dtype=float) / 255


def gray(im):
    return im @ np.array([.299, .587, .114]) if im.ndim == 3 else im


def conv1(a, k, axis):
    r = len(k)//2
    pads = [(0, 0)]*a.ndim
    pads[axis] = (r, r)
    padded = np.pad(a, pads, mode='reflect')
    out = np.zeros_like(a, dtype=float)
    for i, w in enumerate(k):
        sl = [slice(None)]*a.ndim
        sl[axis] = slice(i, i+a.shape[axis])
        out += w*padded[tuple(sl)]
    return out


def blur(a, sigma):
    r = int(np.ceil(5*sigma))
    x = np.arange(-r, r+1)
    k = np.exp(-x*x/(2*sigma*sigma)); k /= k.sum()
    return conv1(conv1(a, k, 0), k, 1)


def gradient(a):
    # Correlation kernel gives positive derivative on increasing ramps.
    return conv1(a, [-.5, 0, .5], 1), conv1(a, [-.5, 0, .5], 0)


def laplace(a):
    return conv1(a, [1, -2, 1], 0) + conv1(a, [1, -2, 1], 1)


def boundary(mask):
    b = np.zeros(mask.shape, bool)
    b[1:] |= mask[1:] != mask[:-1]; b[:-1] |= mask[1:] != mask[:-1]
    b[:, 1:] |= mask[:, 1:] != mask[:, :-1]; b[:, :-1] |= mask[:, 1:] != mask[:, :-1]
    return b


def segmentation(im, labels):
    means = np.array([im[labels == k].mean(0) if np.any(labels == k) else np.zeros(3) for k in (0, 1)])
    out = means[labels.astype(int)]
    out[boundary(labels)] = [1, .15, .1]
    return out


def save(path, im):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.uint8(np.clip(im, 0, 1)*255+.5)).save(path)


def panels(path, entries, cols=3, figsize=None):
    rows = (len(entries)+cols-1)//cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize or (4*cols, 3.35*rows), squeeze=False)
    for ax, entry in zip(axes.flat, entries):
        title, a = entry[:2]
        ax.imshow(a, cmap='gray' if a.ndim == 2 else None)
        ax.set_title(title, fontsize=10); ax.axis('off')
        if len(entry)>2: entry[2](ax)
    for ax in list(axes.flat)[len(entries):]: ax.axis('off')
    fig.tight_layout(); fig.savefig(path, dpi=145, bbox_inches='tight'); plt.close(fig)


def metrics(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
