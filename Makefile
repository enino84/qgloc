SCALE ?= smoke
IMAGE ?= qgloc:latest
DOCKER_RUN = docker run --rm -e SCALE=$(SCALE) -e PYTHONHASHSEED=0 \
	-e OMP_NUM_THREADS=1 -v $(PWD)/results:/work/results \
	-v $(PWD)/paper:/work/paper $(IMAGE)

.PHONY: build test smoke quick paper all exp00 exp01 exp02 exp03 shell clean local-test local-all

build:            ; docker build -t $(IMAGE) .
test:             ; $(DOCKER_RUN) python -m pytest tests -q
smoke:            ; $(MAKE) all SCALE=smoke
quick:            ; $(MAKE) all SCALE=quick
paper:            ; $(MAKE) all SCALE=paper
all:              ; $(DOCKER_RUN) bash /work/scripts/run_all.sh $(SCALE)
exp00:            ; $(DOCKER_RUN) python /work/experiments/exp00_setup.py $(SCALE)
exp01:            ; $(DOCKER_RUN) python /work/experiments/exp01_regime.py $(SCALE)
exp02:            ; $(DOCKER_RUN) python /work/experiments/exp02_calibration.py $(SCALE)
exp03:            ; $(DOCKER_RUN) python /work/experiments/exp03_cycled.py $(SCALE)
shell:            ; docker run --rm -it -v $(PWD):/work/src -v $(PWD)/results:/work/results $(IMAGE) bash
clean:            ; rm -rf results/EXP-* results/run_*.log
clean-cache:      ; rm -rf results/cache
local-test:       ; PYTHONPATH=$(PWD) python3 -m pytest tests -q
local-all:        ; PYTHONPATH=$(PWD):$(PWD)/experiments MPLBACKEND=Agg bash scripts/run_all.sh $(SCALE)
