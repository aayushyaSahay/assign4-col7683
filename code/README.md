# Experiment commands and conventions

Run from `code/`:

```sh
python3 q1_scissors.py
python3 q2_regions.py
python3 q3_keypoints.py
python3 tests.py
```

Each question accepts `--image`, `--size` (maximum image dimension), and `--outdir`.
Defaults are Berkeley 35070 at 320 pixels for Q1/Q2, and 118035 at 400 for Q3.
Q1/Q2 annotations refer to the original 481x321 damselfly photograph. Override
`--image` only with a geometrically equivalent image unless you also change the
manually chosen coordinates in the question scripts.

Intensities/colours are in [0,1]. Gaussian convolution is separable with a
5-sigma kernel radius and reflected padding. Gradient and Laplacian use centered
three-tap finite differences. PNGs are display-quantized; algorithms retain float64.

- **Q1:** Marr-Hildreth uses opposite-neighbor sign tests in four directions.
  The slope threshold is the unnormalized difference across those neighbors.
  Distances are linear maps independently normalized for visualization; raw maps
  are saved as `.npy`. Coordinates are `(row,column)` except source annotations
  specified as `(x,y)` in the scripts. Positive symmetric link costs include
  Euclidean step lengths and an epsilon of 1e-5. The manually chosen closed
  object is the large dark wing, not the entire damselfly.
- **Q2:** full RGB covariances in (a)-(c) use MLE (divide by N) with diagonal ridge 4e-4.
  ML uses equal priors. Both likelihoods are individually peak-normalized for
  display. K-means fixes the FG/BG annotations; WCSS is measured after each mean
  update. Graph cut uses 8-neighbor contrast-sensitive Potts penalties, rounds
  capacities at scale 256, and fixes marked pixels using proven local hard bounds.
  Source-side residual reachability is FG. Q2(d) discards those masks, initializes
  Gaussian models from a rectangle interior and 4-pixel inner ring, and holds the
  exterior BG while alternating ML labels and Gaussian fits. The initial ring
  can change class after initialization; (d) uses a stronger covariance ridge
  .008 to avoid fitting a broad foreground mixture of insect and leaf. It follows the
  simplified method requested in (d), with no graph cut or GMM in that subpart.
- **Q3:** scale-normalized LoG, not a library SIFT/DoG routine; 11 log-spaced
  scales. Strict 26-neighbor extrema exclude endpoints. Taylor refinement uses
  log-scale index and pixel coordinates, relocates at most five times, rejects
  non-maxima/ill-conditioned candidates, and deduplicates converged cells.
  A 3-sigma support margin rejects border/fill artifacts. Circle radius is
  sqrt(2)*sigma. SIFT edge rejection uses the spatial Hessian trace^2/determinant,
  not raw Harris magnitude. Harris is demonstrated independently in (b).
  The transformed-image experiment tests geometry without descriptors, using
  one-to-one matches within 3 pixels and a sqrt(2) scale ratio on visible supports.

All selected parameters, convergence counts, model covariances and repeatability
measurements are saved in `outputs/qN/metrics.json`; report tables read these files.
The experiments are deterministic, including noise (seed 4).
