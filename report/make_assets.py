"""Construct report figures and measured text from saved experimental outputs."""
from pathlib import Path
import json
import shutil

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'report';FIGS=REPORT/'figs';FIGS.mkdir(exist_ok=True)
for i in [1,2,3]:
    for p in (ROOT/f'code/outputs/q{i}').glob('*.png'):
        shutil.copy2(p,FIGS/f'q{i}_{p.name}')
a,b,c=[json.loads((ROOT/f'code/outputs/q{i}/metrics.json').read_text()) for i in [1,2,3]]
parts=[]
def add(s):parts.append(s.strip()+'\n')
def table(headers,rows):
    add('| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows))
def fig(name,caption,width='100%'):
    add(f'![{caption}](figs/{name}){{width={width}}}')

add(r'''
\newpage
# 1. Intelligent Scissors

## (a) Marr--Hildreth edges

For smoothed intensity $L=G_\sigma*I$, form $R=\nabla^2L$. A pixel is an
edge if some opposite neighbor pair $p-d,p+d$ has opposite signs in $R$ and
$|R(p+d)-R(p-d)|>\tau$. We test horizontal, vertical and both diagonal pairs,
and take the largest qualifying difference as the crossing strength. Exact
zeros are allowed; the strictly positive strength threshold removes flat zeros.
The slope threshold is measured in grayscale intensity per discrete neighbor
pair, without normalization by image maximum or diagonal spacing.

The initial reference setting is $\sigma=1.4$, $\tau=0.01$.
Section 4 reports a wider search and a cleaner selected edge map. The following controlled
sweeps change one parameter at a time. A smaller smoothing scale retains weak
leaf texture and wing detail; larger smoothing suppresses texture but also
loses fine boundaries. Lowering the threshold admits clutter; raising it leaves
gaps and suppresses weaker appendages. Thus stronger smoothing and a larger
threshold both reduce detail, through different mechanisms.
''')
table(['Gaussian sigma','Slope threshold','Detected edge pixels'],[(r['sigma'],r['threshold'],r['edge_pixels']) for r in a['edge_sweeps']])
fig('q1_edges.png','Original with manually selected wing seeds, and the smoothing/threshold sweeps. Seed numbers indicate their traversal order.')
add(r'''
\newpage
## (b) Pixel graph and Dijkstra distance functions

The graph contains all pixels and their eight adjacent neighbors. Let $m(p)$
be the gradient magnitude of $L$, $g(p)=m(p)/\max m$, $z(p)$ the binary
Marr--Hildreth edge map, and $t(p)$ a unit vector tangent to the gradient
(normal rotated by 90 degrees). For step direction $e=(q-p)/\|q-p\|$, define

$$f_Z(p,q)=1-\frac{z(p)+z(q)}2,$$
$$f_G(p,q)=1-\frac{g(p)^\gamma+g(q)^\gamma}2,$$
$$f_D(p,q)=\frac{\arccos|t(p)\cdot e|+\arccos|t(q)\cdot e|}{\pi}.$$

A zero gradient has zero tangent and therefore contributes $\pi/2$ to the
angle, rather than an arbitrary preferred direction. Absolute dot products
make the direction term independent of edge polarity. The link cost is

$$w(p,q)=\|q-p\|(w_Zf_Z+w_Gf_G+w_Df_D+10^{-5}).$$

These costs are symmetric and positive. The Euclidean factor accounts for
diagonal steps; the epsilon prevents zero-cost plateaus. This is an explicit
symmetric Intelligent Scissors variant. Baseline weights are
$(0.43,0.43,0.14)$ and $\gamma=1$. We implement heap-based Dijkstra, relaxing
all eight neighbors and keeping predecessors. Complexity is
$O(HW\log(HW))$ time and $O(HW)$ storage for this bounded-degree graph.

The four runs use $(1,0,0)$, $(0,1,0)$, $(0,0,1)$ and the baseline weights.
All start at seed 1. Each distance image is divided by its own maximum;
brightness scales therefore cannot be compared numerically between panels.
Unnormalized distance arrays are retained in the output directory.
''')
table(['Cost term','Maximum distance','Mean distance'],[(k,f"{v['max']:.3f}",f"{v['mean']:.3f}") for k,v in a['distance_maps'].items()])
fig('q1_distances.png','Dijkstra distances from the first seed; dark means cheap to reach. Maps are independently linearly normalized.')
add(r'''
Zero crossings favor travel along binary edge chains but provide little
preference within edge-free regions. Gradient magnitude provides a graded
preference for strong boundaries. Orientation favors travel tangent to local
edges even when their contrast is weak, and can follow irrelevant texture.
The combined metric balances these cues; none encodes object semantics.

## (c) Closed paths and two changes to the link costs

We manually choose eleven seeds around the **large dark wing**, following the
order shown in (a). The selected object is this wing silhouette rather than
the entire insect. For each consecutive seed pair, including the last to the
first, we run Dijkstra and trace predecessors. Segment endpoints are shared,
and the final point equals the first, yielding a closed path.
''')
add('The stored `(row, column)` seeds at the processed resolution are: '+', '.join(f'`({y}, {x})`' for y,x in a['seeds_yx'])+'.')
add(r'''
Two modifications use exactly the same seeds: (1) increase orientation weight
to $(0.2,0.2,0.6)$, to favor tangential continuity; (2) change the magnitude
term to $1-(g(p)^2+g(q)^2)/2$ and weights to $(0.2,0.7,0.1)$, to penalize
weak gradients more strongly. They require no new edge-detector library.
''')
table(['Variant','Path steps','Own metric cost','Pixel-set Jaccard vs baseline'],[(k,v['path_pixels'],f"{v['cost']:.3f}",f"{v['overlap_with_baseline']:.3f}") for k,v in a['paths'].items()])
fig('q1_paths.png','Closed wing contours: yellow paths and red seeds. Weights and gradient exponent are the only changes.')
add('All three paths track the broad wing silhouette. Orientation emphasis moves the path along nearby parallel edge evidence; the stronger gradient variant stays much closer to the baseline. The path overlaps quantify these changes, not accuracy against a human mask. Costs belong to different weight functions and cannot rank segmentation quality. Seed placement constrains the result strongly, and this demonstration does not recover the insect tail or legs.')

add(r'''
\newpage
# 2. User-guided region segmentation

## (a) Full-covariance Gaussian maximum likelihood

We mark two foreground rectangles in the dark wing and four background
rectangles in the surrounding scene; the diagram shows their exact coverage.
Fit one three-dimensional Gaussian to each class, with equal priors:

$$\mu_k=\frac1{N_k}\sum_i c_i,\qquad
\Sigma_k=\frac1{N_k}\sum_i(c_i-\mu_k)(c_i-\mu_k)^T+0.0004\,I_3.$$

The ridge stabilizes inversion for the nearly uniform dark wing. It is a
specified regularized covariance estimate, rather than an unregularized MLE.
The regional cost is the full negative log density

$$D_k(c)=\tfrac12[(c-\mu_k)^T\Sigma_k^{-1}(c-\mu_k)+\log\det\Sigma_k+3\log(2\pi)].$$

Assign the class with the smaller $D_k$. Using the determinant term matters:
a broad background model should not automatically win by having small
Mahalanobis distances. Likelihood displays are $\exp(-D_k)$ divided by that
class's peak value, so each panel shows shape within a class, not comparable
absolute brightness between classes. The classifier itself uses the original
log likelihoods. The segmentation replaces each pixel by its assigned class's
empirical mean and colors pixels on both sides of any four-neighbor label
boundary red. Q2(a) itself does not forcibly override labels at marked pixels.
''')
add(f"Training uses {b['foreground_training_pixels']:,} FG and {b['background_training_pixels']:,} BG pixels. ML assigns {b['ml_foreground_pixels']:,} pixels to foreground.")
table(['Class','Mean R','Mean G','Mean B'],[(name,*[f'{x:.4f}' for x in model['mean']]) for name,model in zip(['Background','Foreground'],b['models'])])
fig('q2_likelihood.png','Marked class pixels, separate peak-normalized likelihoods, and the maximum-likelihood segmentation.')
add('The dark wing separates well from green scenery. A single dark-color Gaussian misses some bright blue body pixels and fine legs, and isolated dark background pixels can be selected. This is a colour-model limitation rather than evidence of complete semantic insect segmentation. Full covariance matrices and source-coordinate rectangles are recorded in the metrics JSON.')
add(r'''
\newpage
## (b) Two-class constrained k-means

Initialize the two means from the same marked pixels. Alternate nearest-mean
assignment in Euclidean RGB distance and recomputation of each mean. Keep
known FG/BG pixels fixed as permitted by the question. The plotted objective is
$\mathrm{WCSS}=\sum_p\|c_p-\mu_{l_p}\|^2$, evaluated after updating means.
A constrained assignment minimizes it over permitted labels; the mean update
minimizes it for those labels. Hence each iteration is non-increasing.
Terminate when all labels stop changing, with a safety cap of 100 iterations.
''')
add(f"The run converges in {len(b['kmeans_wcss'])} iterations. WCSS decreases from {b['kmeans_wcss'][0]:.6f} to {b['kmeans_wcss'][-1]:.6f}; the largest successive increase is {max(0,max(y-x for x,y in zip(b['kmeans_wcss'],b['kmeans_wcss'][1:]))):.2g}.")
fig('q2_kmeans.png','First three k-means segmentations and the converged result, with class mean colors and red boundaries.','80%')
fig('q2_wcss.png','Measured WCSS after each assignment/mean-update iteration.','60%')
add('K-means expands the dark foreground beyond the Gaussian ML result because it uses spherical equal-distance classes. Its marked pixels remain correct by construction, but it does not model the strong correlations or different spreads of the two classes. Convergence of WCSS does not establish correctness against the intended object.')
add(r'''
\newpage
## (c) Contrast-sensitive binary graph cut

Using the fixed Gaussians from (a), minimize

$$E(l)=\sum_p\widetilde D_{l_p}(c_p)+
\lambda\sum_{\{p,q\}\in\mathcal N_8}
\frac{\exp[-\|c_p-c_q\|^2/(2\sigma^2)]}{\|p-q\|}[l_p\ne l_q],$$

where $\widetilde D_k=D_k-\min_jD_j$ is nonnegative and subtracts only a
label-independent per-pixel constant. Each undirected neighbor pair is counted
once. Similar colors strongly resist being separated; dissimilar colors allow
cuts at lower cost. RGB differences and $\sigma$ use [0,1] colour units.

Make source-side pixels FG. A source-to-pixel edge has capacity
$\widetilde D_0$, and a pixel-to-sink edge has capacity $\widetilde D_1$.
Two directed neighbor arcs encode each Potts term. Multiply capacities by 256
and round to integers for SciPy's solver [2]. A cut is exact for that quantized
energy; each finite capacity has rounding error at most $1/512$ in original
energy units. We do not claim exact minimization of unquantized real costs.

To enforce an annotation at $p$, set its incorrect unary capacity to one more
than the maximum finite unary at $p$ plus the sum of its incident pairwise
capacities. Correcting a wrongly labeled seed saves more than any increase in
its other incident costs. Thus an optimum respects every seed. Bounds are
checked before conversion to the solver's capacity range. After maximum flow,
our queue traversal from the source follows strictly positive residual
capacities in $C-F$; reachable image pixels are FG. Seed enforcement is checked
for every reported run.

The reference baseline is $\lambda=2$, $\sigma=0.08$. The sweeps change one
parameter at a time: $\lambda=0.02,40$ and $\sigma=0.008,0.8$.
''')
table(['Lambda','Sigma','FG pixels','Real shifted energy','Integer min cut'],[(r['lambda'],r['sigma'],r['foreground_pixels'],f"{r['energy']:.3f}",r['integer_cut']) for r in b['graphcuts']])
fig('q2_graphcuts.png','Baseline graph cut and much smaller/larger smoothness and color-scale settings.','78%')
add('Lower lambda gives a result close to independent ML, apart from fixed annotations. Larger lambda suppresses speckles and removes some thin structures. Very small sigma allows many inexpensive boundaries even within the object, weakening smoothness. Very large sigma makes pairwise weights nearly independent of contrast, favoring smoother contours and erasing narrow appendages. The baseline retains the main silhouette while reducing clutter; Section 4 quantifies the region/boundary tradeoff against stronger smoothing. Some thin bright structures remain absent because the foreground model was learned mainly from the dark wing. Energies from different parameter settings define different objectives and are not directly comparable measures of accuracy.')
add(r'''
\newpage
## (d) Rectangle-only Gaussian/ML alternation

Discard all earlier FG/BG subrectangles. The only annotation is a rectangle
around the insect. Initialize FG from the rectangle interior excluding a
four-pixel inner boundary ring; initialize BG from this ring. All exterior
pixels remain BG; the ring is only an initialization and is free to change.
Use full-covariance Gaussians with a larger diagonal ridge of 0.008.
Trials at 0.0004, 0.001 and 0.002 retained much of the green leaf as foreground;
0.004, 0.008 and 0.016 separated the main dark insect. We select 0.008 to
reduce covariance overfitting from the initially mixed rectangle interior. Alternate ML label
assignment and refitting each Gaussian from all pixels currently assigned to
its class, including the exterior for later BG estimates. Stop at an unchanged
label mask, with a safety cap of 60 iterations.

This implements the Gaussian/maximum-likelihood alternation explicitly requested
in (d). It is a simplified GrabCut-like procedure: there is one Gaussian per
class and no spatial cut in this particular subpart. Q2(c) separately implements
the graph-cut energy. Holding the exterior BG is an explicit rectangle
constraint; none of the previous foreground strokes are used.
''')
add('The processed-image rectangle `(x0,y0,x1,y1)` is `'+str(b['rectangle_xyxy'])+'` (upper bounds excluded).')
table(['Iteration','FG pixels','Changed labels'],[(r['iteration'],r['foreground_pixels'],str(r['changed']) if r['iteration']>1 else 'initial assignment') for r in b['rectangle_iterations']])
fig('q2_rectangle.png','Rectangle input, initialization ring, first three Gaussian/ML iterations, and final mask.')
add('The final labels stop changing, with no iteration cap reached. The initialization contains a large amount of leaf, which the iterations progressively reject. Compared with stroke-trained Gaussian ML, this model captures more of the dark thin legs, but bright blue parts and local errors remain. Color overlap, a single Gaussian per class, and no spatial regularizer limit this method. No global optimality or full GrabCut performance is claimed.')

add(r'''
\newpage
# 3. Scale-space keypoints

## (a) Scale-normalized LoG extrema and subpixel refinement

For each geometric scale $\sigma_s=1.1\,2^{s/2}$, $s=0,\ldots,10$, compute
$R_s=\sigma_s^2\nabla^2(G_{\sigma_s}*I)$. The scale factor prevents a systematic
bias toward small-scale responses. We build the LoG directly, rather than
using the Difference of Gaussians approximation. A candidate is a strict local
maximum of $|R|$ over its 26 neighbors in position and scale. End scales and
outermost image pixels are excluded; the initial contrast floor is 0.005.

For refinement fit a quadratic to $A=|R|$ in coordinates $(s,y,x)$:

$$A(p+\delta)\simeq A(p)+g^T\delta+\tfrac12\delta^TH\delta,
\qquad \delta=-H^{-1}g.$$

Derivatives are centered differences, including mixed differences. Require a
negative-definite Hessian and reject nearly flat fits. If any offset exceeds
half a cell, move to an adjacent cell and refit, for at most five steps. Accept
only converged fits, with interpolated response $A(p)+g^T\delta/2$, and remove
duplicates that converge to the same grid cell. Fractional scale indices map
back exponentially to sigma. For all displayed blob/keypoint sets, exclude a
three-sigma neighborhood crossing an image boundary. Circle radius is
$\sqrt2\sigma$, corresponding to the characteristic LoG blob radius.
''')
add('The sampled sigmas are '+', '.join(f'{s:.3f}' for s in c['sigmas'])+'.')
fig('q3_scale_space.png','Absolute scale-normalized LoG responses at six of the eleven scales; each panel uses its own display range.')
fig('q3_blobs.png','Discrete and Taylor-refined maxima after support filtering. Refinement also rejects unstable extrema, so the point sets differ.')
add(f"There are {c['discrete_maxima']} supported discrete maxima and {c['localized_maxima']} supported localized maxima. The mean distance of a refined position to its nearest original discrete candidate, in `(scale-index,y,x)` units, is {c['mean_localization_shift_index_y_x']:.3f}. This is a shift measurement, not a localization-error estimate.")
add('A separate analytic test fits a quadratic peak centered at `(3.2,4.3,5.15)` and recovers those coordinates to floating-point tolerance, while the integer peak has a nonzero location error. The transformed-image experiment in (d) provides an image-based comparison. Visual smoothness and fewer points alone cannot establish improved accuracy on an unannotated photograph.')
add(r'''
\newpage
## (b) Harris--Stephens corners

At derivative scale 1.0, compute $I_x,I_y$ and the structure tensor

$$M=G_{1.8}*\begin{pmatrix}I_x^2&I_xI_y\\I_xI_y&I_y^2\end{pmatrix},
\qquad C=\det M-0.04(\operatorname{tr}M)^2.$$

Visualize determinant, trace and signed cornerness with signed-log compression
$\operatorname{sign}(v)\log(1+|v|/(10^{-4}\max|v|))$ to expose weaker responses.
Detect positive local maxima in a 7 x 7 neighborhood, above 1.5% of the maximum
positive response. Trace measures overall gradient energy; determinant is
large when both tensor eigenvalues are appreciable. Harris suppresses
one-directional edges through its trace-squared penalty.
''')
fig('q3_harris.png','Structure-tensor determinant, trace, signed Harris response, and detected corners.')
add(f"The detector finds {c['harris_corners']} corners, concentrated around facade and window junctions. Long smooth edges can have large trace but weak or negative Harris response; the sky is mostly suppressed. The fixed-scale corner set is distinct from the multiscale blobs used in (c).")
add(r'''
\newpage
## (c) SIFT contrast and edge-response criteria

Apply the SIFT localization tests [3] to the refined LoG candidates. The
contrast criterion keeps the interpolated absolute response above threshold
$T$. For cornerness, use the spatial 2 x 2 Hessian $H_{xy}$ of the signed
scale-normalized response at the nearest scale-space cell. Keep points with
positive determinant and

$$\frac{(\operatorname{tr}H_{xy})^2}{\det H_{xy}}<\frac{(r+1)^2}{r}.$$

This is SIFT's rejection of anisotropic edge responses, not a threshold on the
Harris image from (b). Smaller $r$ is stricter; a ratio near one means balanced
principal curvatures. The independent contrast sweep effectively disables the
ratio ceiling, while the independent cornerness sweep imposes no additional
contrast beyond the initial candidate floor. We also show their intersection.
''')
table(['Criterion','Surviving localized keypoints'],[(k.replace('_',' '),v) for k,v in c['criteria_counts'].items()])
fig('q3_criteria.png','Independent contrast thresholds (top) and edge-ratio thresholds (bottom).')
fig('q3_selected.png',f"Both selected criteria T=0.025 and r=10: {c['selected_keypoints']} points.",'85%')
add('Raising contrast preferentially removes faint responses. Tightening the curvature ratio removes elongated edge detections, while preserving more balanced blobs. We initially select T=0.025 and r=10 as a visual compromise between coverage and clutter on the architecture. Section 4 tests a wider joint threshold grid. These settings are not optimized against human keypoint annotations or a held-out evaluation set. Large circles represent larger spatial support, not higher confidence.')
add(r'''
\newpage
## (d) Transformation and geometric repeatability

Create a copy using a 22-degree rotation in image coordinates (positive y
downward), scale 0.82 around the image center, intensity map $0.85I+0.07$, and
independent additive Gaussian noise with standard deviation 0.008 in RGB
(seed 4). Bilinear resampling and inverse mapping are implemented from pixel
arrays; transformed image borders are filled with gray. Intensities are clipped
to [0,1] only after the intensity/noise transformation.

Run the same detector and selected criteria on the new image. Transform each
original keypoint center by the known similarity matrix and its sigma by 0.82.
The diagram overlays detected points in cyan and transformed originals in
magenta. Detections whose three-sigma support touches the artificial fill are
excluded. For repeatability, only original points whose transformed support is
fully visible are eligible. A correspondence requires centers within 3 pixels
and a scale ratio within $\sqrt2$; sorted candidate distances are greedily
matched one-to-one. The score is matched pairs / eligible originals. This is
geometric detector repeatability, without descriptors or orientation assignment.
''')
fig('q3_transformed.png','Original selected points and the modified-image comparison, using cyan detections and dashed magenta projected originals.')
d=c['discrete_repeatability'];l=c['repeatability']
table(['Detector','Visible originals','Matches','Repeatability','Mean matched error (px)'],[(name,s['eligible'],s['matches'],f"{s['repeatability']:.3f}",f"{s['mean_error_px']:.3f}" if s['mean_error_px'] is not None else 'n/a') for name,s in [('Discrete',d),('Taylor refined',l)]])
add(f"The modified image has {c['modified_keypoints']} selected supported detections. The refined detector repeats {l['matches']} of {l['eligible']} eligible original points ({100*l['repeatability']:.1f}%).")
add(f"Taylor refinement changes mean matched error from {d['mean_error_px']:.3f} to {l['mean_error_px']:.3f} pixels. Repeatability changes from {100*d['repeatability']:.1f}% to {100*l['repeatability']:.1f}%, so the observed benefit is finer localization of accepted matches rather than a higher matched fraction.")
add('The two detector rows use their respective eligible sets, so they measure overall detector behavior rather than paired error on an identical set. Subpixel fitting can reduce quantization, but may reject unstable fits or alter the set that passes contrast/curvature tests. Gain changes LoG contrast, rotation changes the sampling grid, noise perturbs derivatives, and the finite scale ladder shifts preferred scales. Consequently the correspondence is partial. The plots and measured errors describe the observed result; no perfect scale/rotation invariance is assumed.')
from study_assets import study_sections
add(study_sections(ROOT))

add(r'''
\newpage
# Verification and reproducibility

Fourteen test groups check Gaussian impulse normalization and constants,
polynomial derivatives, step-edge behavior, Dijkstra against independent
Bellman--Ford relaxation and known grid distances, analytic Gaussian log
likelihoods, constrained k-means monotonicity, binary graph cuts against all
64 labelings of a 2 x 3 image across five random examples, rectangle-only separation of two synthetic colors, exact Taylor recovery
of a known subpixel quadratic peak, strict extrema/tie rejection, Harris on
constant/quadrant images, identity resampling/geometric repeatability, and mask-agreement scores with analytically known values.
All tests use small deterministic arrays; experimental masks are not used as
their expected answers. The covariance regularization, hard-seed enforcement,
integer energy quantization and limitations are explicit in the implementation.

`make all` writes `COL7683_A4_code.zip` and `COL7683_A4_report.pdf` in the
submission directory. Source code, report source, original inputs and build
instructions are bundled. Intermediate PNGs, raw arrays and JSON metrics are
regenerated by the build. Each question also runs independently from `code/`.
The PDF includes both team members' names. Each member must separately submit
the contribution response required in the assignment instructions.

# References

[1] UC Berkeley, *The Berkeley Segmentation Dataset and Benchmark*, BSDS300,
training image archive and dataset gallery. Original photographs credited to
Corel. Accessed 14 November 2026.
<https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/>

[2] SciPy documentation, `scipy.sparse.csgraph.maximum_flow`: sparse directed
integer capacities and residual flow. Accessed 14 November 2026.
<https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.maximum_flow.html>

[3] David G. Lowe, *Distinctive Image Features from Scale-Invariant Keypoints*,
International Journal of Computer Vision, 2004. Taylor localization and spatial
Hessian edge-response criterion; this assignment uses direct LoG extrema.
<https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf>

[4] Course lecture 21, *Edge detection*, COL7(6)83: Gaussian/Laplacian filtering
and Marr--Hildreth opposite-neighbor zero crossings. Supplied course material.
''')
(REPORT/'generated.md').write_text('\n'.join(parts))
print('Generated measured report content and figures')
