"""Measured report supplement for the reproducible expanded parameter study."""
import json
import shutil


def study_sections(root):
    output=root/'code/outputs/study';figs=root/'report/figs'
    for path in output.glob('*.png'):shutil.copy2(path,figs/('study_'+path.name))
    load=lambda name:json.loads((output/(name+'.json')).read_text())
    q1,q2,q3=[load(f'q{i}_summary') for i in (1,2,3)]
    pieces=[]
    def add(s):pieces.append(s.strip()+'\n')
    def table(headers,rows):
        add('| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows))
    def fig(name,caption,width='100%'):
        add(f'![{caption}](figs/study_{name}){{width={width}}}')
    def page(title):add('\\newpage\n'+title)
    page('# 4. Expanded parameter study\n\n## Protocol and reference masks')
    add('We expand the initial controlled demonstrations into a reproducible sensitivity study. Algorithms, images, processing resolution and manual seeds stay fixed while parameters vary. The original runs remain reference baselines; the improved settings and their tradeoffs are reported here. Every trial is recorded in CSV/JSON files under `code/outputs/study`, also bundled in the ZIP. `make study` reproduces the study; `make all` includes it.')
    table(['Experiment','Settings evaluated'],[
        ['Marr--Hildreth grid',q1['edge_runs']],['Closed-path settings / seed ablation',q1['path_runs']],
        ['Gaussian ML / k-means variants',q2['model_runs']],['Graph-cut joint grid',q2['graphcut_runs']],
        ['Rectangle ridge / ring-width grid',q2['rectangle_runs']],['Keypoint contrast / curvature pairs',q3['threshold_pairs']],
        ['Threshold pairs x transformation cases',q3['threshold_transform_runs']],['Harris settings',q3['harris_runs']],['Scale ladders',len(q3['scale_ladders'])]])
    add(r'''For image 35070, use all five supplied color segmentations from the BSDS300 human archive [1], annotators 1105, 1108, 1109, 1112 and 1124. Region 1 corresponds to the main insect in each annotation. Resize label maps by nearest-neighbor sampling, then evaluate predictions separately against each reference and average the five scores. Do not choose a favorable annotator. The reference masks never enter model fitting, edge costs, seed placement or cut construction.

Foreground intersection-over-union is $|P\cap G|/|P\cup G|$. Pixel precision/recall are also recorded. Boundary F1 uses the boundary convention in Q2(a), with a two-pixel Euclidean tolerance in both directions. This tolerance-based score permits multiple nearby pixels to match; it is a transparent diagnostic, not the official BSDS boundary benchmark. The unmarked IoU excludes all earlier FG/BG training rectangles from both intersection and union. It checks behavior away from the supplied strokes, but is still measured on the same image.

These are development-image scores used to compare settings. Five annotations describe ambiguity in one scene; they are not five independent photographs or a held-out test set. No significance interval or universal optimality claim follows from them.''')
    fig('q2_references.png','Input and all five human insect-region references. Only the evaluation code reads these annotations.','80%')

    page('## 4.1 Edge scale and slope threshold')
    add('The joint grid uses Gaussian sigma {0.4, 0.6, 0.8, 1, 1.4, 2, 3, 4} and slope threshold {0.001, 0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.12, 0.16, 0.24, 0.32}. Evaluate against boundaries of the full human label maps, since generic edges include leaf boundaries as well as the insect. Human maps omit some genuine texture, so disagreement is not necessarily a detector implementation error.')
    best=q1['best_edge'];base=q1['baseline_edge']
    table(['Setting','Sigma','Threshold','Precision','Recall','Boundary F1'],[(n,r['sigma'],r['threshold'],f"{r['precision']:.3f}",f"{r['recall']:.3f}",f"{r['boundary_f1']:.3f}") for n,r in [('Initial',base),('Best tested',best)]])
    add(f"The best tested pair is sigma={best['sigma']:g}, threshold={best['threshold']:g}. Its mean boundary F1 is {best['boundary_f1']:.3f}, compared with {base['boundary_f1']:.3f} initially. The original low threshold recovered almost all annotated boundaries but admitted substantial clutter. The new pair sacrifices a little recall for much higher precision. It lies inside the expanded grid rather than at its endpoint. We recommend this pair for the final generic edge map; the Q1 distance/path demonstrations retain their stated common reference setting to isolate cost-term effects.")
    fig('q1_edges_grid.png','Joint sensitivity of boundary F1; high score means better agreement with the five human maps.','93%')
    fig('q1_edges_selected.png','Initial edge map versus the cleaner setting selected by the expanded grid.','95%')

    page('## 4.2 Closed-path stability and graph-cut tradeoffs')
    paths=load('q1_paths');best_path=max(paths,key=lambda r:r['human_boundary_support']);worst_path=min(paths,key=lambda r:r['human_boundary_support']);sparse=paths[-1]
    add('For paths, test all 15 nonnegative weight triples summing to one in increments of 0.25, the original weights, three gradient exponents, six edge-scale/threshold changes, and a six-seed ablation (every other original seed). All 26 runs close successfully. Boundary support is the fraction of path pixels within two pixels of a human insect boundary, averaged across annotators. This is a support/precision diagnostic, not whole-object recall: the target is only the wing, and the seeds already constrain its location. Jaccard measures agreement with the original path rather than correctness.')
    table(['Run','Weights (Z,G,D)','Seeds','Steps','Boundary support','Baseline Jaccard'],[(n,str(r['weights']),r['seed_count'],r['steps'],f"{r['human_boundary_support']:.3f}",f"{r['baseline_jaccard']:.3f}") for n,r in [('Original',paths[0]),('Highest support',best_path),('Lowest support',worst_path),('Sparse seeds',sparse)]])
    add('Different metrics can choose nearby parallel boundaries even with identical seeds. More support alone does not prove a better closed object contour, and own-metric path costs cannot rank different weight functions. The original combined contour remains the illustrated balanced reference; the full records expose its sensitivity rather than asserting unique best weights.')
    add('For graph cuts, evaluate lambda {0.02, 0.1, 0.5, 2, 10, 40, 160} jointly with color sigma {0.008, 0.02, 0.04, 0.08, 0.16, 0.4, 0.8}. All 49 cuts respect the marked FG/BG pixels. The Gaussian models remain fixed from Q2(a), so this sweep isolates pairwise penalties.')
    table(['Setting','Lambda','Sigma','Mean IoU','Unmarked IoU','Boundary F1'],[(n,r['lambda'],r['sigma'],f"{r['iou']:.3f}",f"{r['unmarked_iou']:.3f}",f"{r['boundary_f1']:.3f}") for n,r in [('Original',q2['baseline_graphcut']),('Highest IoU',q2['best_graphcut'])]])
    add(f"The highest mean IoU is {q2['best_graphcut']['iou']:.3f} at lambda={q2['best_graphcut']['lambda']:g}, sigma={q2['best_graphcut']['sigma']:g}; individual-annotator IoUs range from {q2['best_graphcut']['iou_min']:.3f} to {q2['best_graphcut']['iou_max']:.3f}. This improves area overlap but lowers boundary F1 relative to lambda=2. Retain both results: lambda=40 is the area-overlap choice, while lambda=2 better preserves fine boundary agreement. Increasing both parameters too far erases most of the insect, as the low-score corner of the grid shows.")
    fig('q2_graphcut_grid.png','Joint lambda/sigma study. Stronger smoothing is not uniformly better.','87%')

    page('## 4.3 Region-model and rectangle initialization sensitivity')
    models=load('q2_models');best_ml=max((r for r in models if r['method']=='Gaussian ML'),key=lambda r:r['iou']);base_ml=next(r for r in models if r.get('regularization')==.0004)
    add('Test Gaussian covariance ridges {0.0001, 0.0004, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032}. Also compare k-means with and without fixed annotated pixels, using exactly the prescribed initial class means. Both runs converge with non-increasing WCSS. The two WCSS values optimize different feasible label sets and cannot by themselves establish which segmentation is better.')
    table(['Model','Ridge / constraint','Mean IoU','Unmarked IoU','Boundary F1'],[(name,str(param),f"{r['iou']:.3f}",f"{r['unmarked_iou']:.3f}",f"{r['boundary_f1']:.3f}") for name,param,r in [('Gaussian initial',.0004,base_ml),('Gaussian highest IoU',best_ml['regularization'],best_ml)]+[('K-means',r['constrained'],r) for r in models if r['method']=='K-means']])
    add('For rectangle-only alternation, cross the eight covariance ridges with inner-ring widths {2, 4, 8, 12} pixels, keeping the enclosing rectangle fixed. The ring supplies initial background colors only; its labels may subsequently change. All 32 runs stop at an unchanged mask before the 60-iteration cap. The wider ring changes the initial color distribution as well as the size of the initial interior.')
    table(['Rectangle setting','Ridge','Ring width','Iterations','Mean IoU','Boundary F1'],[(n,r['regularization'],r['ring_width'],r['iterations'],f"{r['iou']:.3f}",f"{r['boundary_f1']:.3f}") for n,r in [('Original',q2['baseline_rectangle']),('Highest IoU',q2['best_rectangle'])]])
    fig('q2_rectangle_examples.png','A weak regularization failure, the original rectangle setting, and the improved setting.','100%')
    add(f"The improved rectangle setting has ridge {q2['best_rectangle']['regularization']:g} and ring width {q2['best_rectangle']['ring_width']}, raising mean IoU from {q2['baseline_rectangle']['iou']:.3f} to {q2['best_rectangle']['iou']:.3f} and boundary F1 from {q2['baseline_rectangle']['boundary_f1']:.3f} to {q2['best_rectangle']['boundary_f1']:.3f}. We recommend this as the tuned rectangle result, while retaining the initial iteration sequence in Q2(d). Some background islands and bright body errors remain. The improvement is specific to this image and annotation protocol, not a guarantee for an arbitrary rectangle or photograph.")

    page('## 4.4 Joint keypoint thresholds and separate perturbations')
    add('Cross contrast thresholds {0.008, 0.015, 0.025, 0.04, 0.06, 0.08, 0.12, 0.18} with edge ratios {2, 3, 5, 8, 10, 15, 20, 30}. Apply all 64 pairs to nine transformation cases, for 576 evaluations. Compute scale spaces once per image; each threshold pair filters the same candidate set, so the comparison is controlled. Average the combined-transform score across noise seeds 4, 17 and 29. Do not treat deterministic parameter trials as independent statistical replicates.')
    base=q3['baseline_threshold'];best=q3['best_repeatability_with_100_points']
    table(['Threshold choice','T','r','Original points','Mean matches','Mean repeatability'],[(n,r['contrast'],r['edge_ratio'],r['original_count'],f"{r['mean_matches']:.1f}",f"{r['mean_repeatability']:.3f}") for n,r in [('Original',base),('Highest fraction, >=100 points',best)]])
    add('The second row maximizes repeatability among settings leaving at least 100 original points, an explicit coverage constraint. A higher fraction comes with fewer accepted points and fewer actual matches. Therefore the original T=0.025, r=10 remains a useful coverage-oriented choice, while T=0.04, r=5 is a stricter repeatability-oriented alternative; neither is unconditionally best.')
    fig('q3_repeatability_grid.png','Mean repeatability across the three combined-transform noise seeds.','85%')
    transforms=load('q3_transforms');rows=[]
    for name in dict.fromkeys(r['transform'] for r in transforms):
        d=next(r for r in transforms if r['transform']==name and r['method']=='Discrete');l=next(r for r in transforms if r['transform']==name and r['method']=='Localized')
        rows.append((name,f"{d['repeatability']:.3f}",f"{l['repeatability']:.3f}",l['matches'],f"{l['mean_error_px']:.3f}"))
    table(['Perturbation','Discrete repeatability','Refined repeatability','Refined matches','Error (px)'],rows)
    add('Rotation-only is 22 degrees; scale-only is 0.82; intensity-only is 0.85I+0.07; noise-only has RGB standard deviation 0.008. Combined cases use all four. The harder case uses 37 degrees, scale 0.65, gain 0.7, offset 0.1 and noise 0.016. Identity reproduces every eligible point after correcting bilinear sampling at the final image row/column. Intensity and noise are relatively mild; spatial resampling reduces repeatability much more. The three combined seeds show the result is not dependent on a single noise draw.')

    page('## 4.5 Paired localization errors, scale density and Harris stability')
    add('To compare position errors on the same features, retain only scale-space cells present in both discrete and refined selections in both images. Form the same one-to-one correspondences for both methods, requiring both their errors to be within 3 pixels and both scale ratios within sqrt(2). Pair candidates are sorted by their mean discrete/refined error. This symmetric shared subset avoids comparing different accepted point sets, but still conditions on jointly stable peaks and is not a keypoint ground-truth dataset.')
    paired=load('q3_paired')
    table(['Perturbation','Shared pairs','Discrete error (px)','Refined error (px)'],[(r['transform'],r['pairs'],f"{r['discrete_error']:.3f}",f"{r['localized_error']:.3f}") for r in paired])
    add('Refinement lowers shared-pair position errors for rotation, scaling and all combined cases. For noise-only, unchanged integer cells can have zero error, while the refined positions move slightly; refinement is not uniformly beneficial for every perturbation. Identity errors are numerical roundoff. Thus the supported conclusion is reduced quantization error under spatial transforms, not universal improvement in detector repeatability.')
    table(['Scales per octave','Scale images','Original points','Matches','Repeatability','Error (px)'],[(r['scales_per_octave'],r['scale_count'],r['original_count'],r['matches'],f"{r['repeatability']:.3f}",f"{r['mean_error_px']:.3f}") for r in q3['scale_ladders']])
    add('The 2-, 3- and 4-samples-per-octave ladders span the same sigma range 1.1 to 35.2 and use the same T=0.025, r=10. Denser ladders lower matched-position error on the seed-4 combined case, but repeatability is not monotonic. Three samples per octave is a useful alternative if finer scale sampling is preferred; a denser ladder alone does not establish invariance. These scores again use each detector\'s respective accepted set.')
    harris=load('q3_harris')
    add('The 81 Harris settings cross derivative sigma {0.6, 1, 1.6}, tensor-window sigma {1, 1.8, 3}, k {0.03, 0.04, 0.06}, and relative threshold {0.005, 0.015, 0.04}, always using 7x7 nonmaximum suppression. The table summarizes counts over the other settings for each derivative scale. There is no human corner annotation here: counts establish sensitivity, not corner-detection accuracy.')
    table(['Derivative sigma','Minimum corners','Maximum corners'],[(s,min(r['corners'] for r in harris if r['derivative_sigma']==s),max(r['corners'] for r in harris if r['derivative_sigma']==s)) for s in [.6,1.,1.6]])
    add('The expanded study supports more precise, qualified claims: generic edge clutter can be substantially reduced; region overlap and boundary preservation trade off; rectangle initialization matters; and subpixel fitting improves location estimates under spatial resampling without guaranteeing a higher repeatability fraction. The report keeps the original demonstrations and exposes better alternatives rather than presenting a single setting as universally optimal.')
    return '\n'.join(pieces)
