PY ?= python3
PDFENGINE ?= pdflatex
export OPENBLAS_NUM_THREADS := 1
export MPLCONFIGDIR := /tmp/dia4-matplotlib
SOURCES := $(wildcard code/dia/*.py) $(wildcard code/q*.py)
INPUTS := code/inputs/35070.jpg code/inputs/118035.jpg

.PHONY: all inputs results figures report submission test clean
all: submission
inputs: $(INPUTS)
results: code/outputs/.stamp
code/outputs/.stamp: $(SOURCES) $(INPUTS)
	cd code && $(PY) q1_scissors.py
	cd code && $(PY) q2_regions.py
	cd code && $(PY) q3_keypoints.py
	@touch $@
figures: report/figs/.stamp
report/figs/.stamp: code/outputs/.stamp report/make_assets.py
	$(PY) report/make_assets.py
	@touch $@
report: report/report.pdf
report/report.pdf: report/report.md report/figs/.stamp
	cd report && pandoc report.md generated.md -o report.pdf --pdf-engine=$(PDFENGINE)
submission: report/report.pdf
	$(PY) code/package_submission.py
test:
	$(PY) code/tests.py
clean:
	$(PY) code/clean.py
