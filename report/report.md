---
title: "COL7(6)83 -- Assignment 4"
subtitle: "Edges, segmentation, and scale-space keypoints"
author:
  - "Aayushya Sahay (2023CS10543)"
  - "Manvendra Rajpurohit (2023CS10936)"
date: "November 2026"
geometry: "a4paper,margin=2cm"
fontsize: 10pt
colorlinks: true
header-includes:
  - \usepackage{float}
  - \floatplacement{figure}{H}
  - \usepackage{booktabs}
  - \usepackage{microtype}
---

# Experimental setup

The supplied PDF is labelled Assignment 3 in error; this is the solution for
**Assignment 4**, due 17 November 2026. We implement the required operations on
pixel arrays. NumPy supplies array arithmetic and small-matrix linear algebra;
Pillow supplies image I/O and the explicitly permitted resizing. Matplotlib is
used only for plots. SciPy supplies sparse matrix storage and the explicitly
permitted maximum-flow solver in Q2(c). We implement the graph construction and
residual cut extraction ourselves. No library filtering, edge detector,
segmentation, shortest-path, clustering, corner or SIFT function is used.

Both source images belong to the **200-image training split of BSDS300** [1].
Image 35070 shows a damselfly on a green leaf: the dark wing supplies a clear
closed contour in Q1, while its body, wing and thin appendages give Q2 a more
challenging foreground. Image 118035 shows a church whose dome, windows and
facade provide blobs and corners for Q3. Original archive paths, source links
and original JPEGs are included in `code/inputs`. Q1 and Q2 use 320 x 214 pixels;
Q3 uses 400 x 267. The brief requires dataset images and does not require any
student-captured photograph. We use visual interpretation of the photographs;
the expanded study compares outputs to all five supplied human segmentations.
These are same-image development measurements, not a held-out benchmark.

Colours are RGB in [0,1]. Grayscale is $0.299R+0.587G+0.114B$. All numerical
processing uses floating point; only final PNG display images are quantized.
Gaussian smoothing uses a normalized separable kernel, radius
$\lceil5\sigma\rceil$, and reflected padding. This larger support avoids the
small truncation ripples that a shorter kernel can amplify after differentiation.
Spatial derivatives use centered differences; the Laplacian is the sum of the
horizontal and vertical $[1,-2,1]$ differences.

`make test` checks independent small references and numerical invariants.
`make all` runs all experiments and creates measured report tables, figures, the expanded parameter study,
this PDF, and the source ZIP. Inputs, annotations, random seeds and parameters
are included, so a clean extraction needs no downloads. The generated sections
below include the results and their interpretation for every subpart.
