# Reproducing Wainwright (2009) - simulation pipeline
#   make reproduce 			everything from scratch: setup -> clean -> test -> all
#   make setup     			create venv/ and install requirements.txt
#   make all       			run simulation -> summaries -> figures
#   make simulate  			run simulations and save raw results
#   make analyze   			summarize raw results
#   make figures   			make all figures
#   make test      			run test suite
#   make clean     			remove generated files
#   make profile			cProfile + line profile, baseline and optimized (10 reps per condition)
#   make complexity       	time one replicate vs n and vs p, baseline and optimized
#   make benchmark        	total time baseline vs optimized, and speedup vs number of workers
#   make parallel         	full study with the optimized code on all cores
#   make stability-check  	warnings, convergence, conditioning, singular X_S (baseline and optimized)
#   make perf-figures   	performance figures from the complexity and benchmark results

# Use the project venv if it exists, otherwise whatever python3 is on PATH
PYTHON ?= $(if $(wildcard venv/bin/python),venv/bin/python,python3)
# Interpreter used once to create the venv
BOOTSTRAP_PYTHON ?= python3

RAW      := results/raw/simulation_raw.csv
SUMMARY  := results/summary.csv
FIG1     := results/figures/fig1_success_vs_n_and_theta.png
PROF       := results/profiling
BENCH_REPS ?= 20

.PHONY: reproduce setup all simulate analyze figures test clean profile complexity benchmark parallel stability-check perf-figures

# One command for a full from-scratch reproduction. Sub-makes run in sequence, so the steps
# after setup use the venv it creates (even on a fresh clone without venv/)
reproduce:
	$(MAKE) setup
	$(MAKE) clean
	$(MAKE) test
	$(MAKE) all

# Create the virtual environment (if missing) and install dependencies
setup:
	test -d venv || $(BOOTSTRAP_PYTHON) -m venv venv
	venv/bin/python -m pip install --upgrade pip
	venv/bin/python -m pip install -r requirements.txt

# Run complete simulation pipeline and generate all outputs
all: figures perf-figures

# Run the simulation and save raw results
simulate: $(RAW)

$(RAW): src/simulation.py src/dgps.py src/methods.py
	$(PYTHON) -m src.simulation --impl optimized --n-jobs -1

# Based on the raw results, make summary statistics
analyze: $(SUMMARY)

$(SUMMARY): $(RAW) src/metrics.py data/paper_values.csv
	$(PYTHON) -m src.metrics

# Make the figures from the summary statistics
figures: $(FIG1)

$(FIG1): $(SUMMARY) src/figures.py
	$(PYTHON) -m src.figures --check pdw_fast

# Run tests
test:
	$(PYTHON) -m pytest

# Remove the generated files (keeps the .gitkeep placeholders)
clean:
	rm -f results/raw/*.csv results/raw/*.json
	rm -f results/summary.csv results/thresholds.csv results/collapse.csv results/paper_comparison.csv
	rm -f results/theory_bounds.csv
	rm -f results/figures/*.png results/figures/*.pdf
	rm -rf .pytest_cache
	find . -name "__pycache__" -type d -not -path "./venv/*" -prune -exec rm -rf {} +

# ---- Unit 3: performance ----
profile:
	mkdir -p $(PROF)
	$(PYTHON) -m src.profile_sim --impl baseline           > $(PROF)/baseline_cprofile.txt
	$(PYTHON) -m src.profile_sim --impl baseline --lines   > $(PROF)/baseline_lines.txt
	$(PYTHON) -m src.profile_sim --impl optimized          > $(PROF)/optimized_cprofile.txt
	$(PYTHON) -m src.profile_sim --impl optimized --lines  > $(PROF)/optimized_lines.txt

complexity:
	$(PYTHON) -m src.complexity --impl baseline
	$(PYTHON) -m src.complexity --impl optimized

benchmark:
	$(PYTHON) -m src.benchmark --n-reps $(BENCH_REPS)

# Always reruns the full study (unlike `make simulate`, which skips it if the results exist)
parallel:
	$(PYTHON) -m src.simulation --impl optimized --n-jobs -1

stability-check:
	mkdir -p $(PROF)
	$(PYTHON) -m src.stability --impl baseline  > $(PROF)/stability_baseline.txt
	$(PYTHON) -m src.stability --impl optimized > $(PROF)/stability_optimized.txt

perf-figures:
	$(PYTHON) -m src.perf_figures --bench-reps $(BENCH_REPS)
	