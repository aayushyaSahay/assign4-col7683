# COL7(6)83 — Assignment 4

Edge detection, interactive and region segmentation, and scale-space keypoints.
The supplied brief mistakenly prints Assignment 3; this repository and submission
use **Assignment 4**, due 17 November 2026.

## Reproduce

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
make test
make all
```

The report needs `pandoc` and `pdflatex`; `make PDFENGINE=xelatex all` is also
supported. A Conda environment is provided. `make all` creates:

- `submission/COL7683_A4_code.zip`
- `submission/COL7683_A4_report.pdf`

`make inputs`, `make results`, `make figures`, `make report`, and `make submission`
are available separately. `make -B results` reruns all experiments. `make clean`
removes only generated files. No downloads, neighboring assignment repositories,
GUI interaction, or manually populated placeholders are needed to reproduce.

## Layout

```
code/dia/                 reusable pixel-array algorithms
code/q1_scissors.py        Marr-Hildreth, Dijkstra, closed paths
code/q2_regions.py         Gaussian ML, k-means, graph cut, rectangle alternation
code/q3_keypoints.py       normalized LoG, Taylor localization, Harris, SIFT tests
code/tests.py             independent references and numerical invariants
code/inputs/              two Berkeley training images and provenance
code/outputs/qN/           generated PNGs, arrays, metrics.json (ignored)
report/report.md          approach, equations and experimental interpretation
report/make_assets.py     figures and measured tables from the actual metrics
report/generated.md      generated report content (ignored)
submission/               two upload files (ignored)
```

Algorithms use NumPy operations on pixel arrays. Pillow provides permitted I/O
and resizing, Matplotlib provides presentation, and SciPy sparse matrices and
`maximum_flow` provide the explicitly permitted max-flow solver. Filtering,
edges, shortest paths, segmentation costs, residual reachability, clustering,
scale-space extrema and keypoint tests are implemented in this repository.

Both inputs are from the required **200-image Berkeley training set**; original
JPGs and source attribution are bundled. **No student-captured photos are required.**
Seeds, marked subrectangles and the single Q2(d) rectangle were manually selected
and recorded in code/metrics; they are user input to the algorithms, not labels
obtained from a segmentation library. See [code/README.md](code/README.md).

The report uses the same team as Assignment 3: Aayushya Sahay (2023CS10543) and
Manvendra Rajpurohit (2023CS10936). Each member must separately submit the
individual contribution form mentioned in the brief. The ZIP/PDF are prepared
locally; Moodle upload and contribution statements are left to the team.

## Expanded parameter study

`make study` evaluates 88 Marr--Hildreth settings, 26 closed-path/seed variants,
49 graph-cut settings, 10 Gaussian/k-means variants, 32 rectangle settings,
64 keypoint threshold pairs across nine perturbation cases (576 evaluations),
81 Harris settings, and three scale ladders. The report includes the sensitivity
maps, improved alternatives, and the tradeoffs behind the recommendations.

The five original Berkeley human annotations are bundled for evaluation only.
The study records mean overlap and boundary agreement across all five; region
models and edge costs never use these labels. These are same-image development
results, not held-out benchmark scores. Keypoints are assessed using the known
transformation, including three noise seeds and paired position-error comparisons.

All trial records are in `code/outputs/study/*.csv` and `*.json`; the ZIP includes
these small tabular records as well as the source needed to regenerate them.
The initial question figures remain controlled reference experiments. The report's
expanded-study section shows the improved edge and rectangle results and the
area/boundary and match-count/repeatability tradeoffs. There is no claim of a
universally optimal parameter setting.
