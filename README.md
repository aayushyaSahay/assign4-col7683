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
