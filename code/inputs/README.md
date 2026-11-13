# Input provenance

Both images are from the **200-image training split of BSDS300**. Downloaded
2026-11-14 from UC Berkeley's official archive:
https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz

- `35070.jpg`: `BSDS300/images/train/35070.jpg`, damselfly on a leaf. Q1 segments
  its closed wing silhouette; Q2 segments the dark insect against green scenery.
- `118035.jpg`: `BSDS300/images/train/118035.jpg`, white church with a red dome.
  Used in Q3 for blobs and architectural corners.

Dataset overview and training gallery:
https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/
https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300/html/dataset/images.html

The archive contains 200 training JPGs; membership was checked from the archive
paths before copying these two files. Original images are retained byte for byte.
The dataset credits the source photographs to Corel. These are dataset images,
not student photographs or a claim of unrestricted public-domain licensing.
Only generated outputs are PNG. No personally captured photos are required by
the assignment, which explicitly requests Berkeley training images in every problem.

## Human references for the parameter study

`human/{1105,1108,1109,1112,1124}.seg` are the five color annotations of image
35070 from the official `BSDS300-human.tgz` archive, downloaded 2026-11-14:
https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-human.tgz
Archive paths: `BSDS300/human/color/<annotator>/35070.seg`.

Each annotator's region **1** identifies the main insect. All five region-1
masks are used, without choosing a favorable annotator. The masks are resized
with nearest-neighbor sampling for evaluation at 320x214; image-derived costs
and the manually selected seeds/rectangles do not read them. Full label maps
are used only for evaluating all-region edge agreement in Q1(a). The five raw
files are included so the study is reproducible without a download. The study
is a same-image development analysis, not a held-out benchmark.
