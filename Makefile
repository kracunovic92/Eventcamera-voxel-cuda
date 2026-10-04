SHELL := /bin/bash
.DEFAULT_GOAL := help

PYTHON ?= .venv/bin/python
PYTHON_FOR_VENV ?= python3
PIP := $(PYTHON) -m pip
CMAKE_PRESET ?= local-debug
BUILD_DIR ?= build/$(CMAKE_PRESET)
BENCH_DIR := build/cpu-benchmark-release

# All paths follow the selected recording and window. Advanced users may
# override any one of these variables explicitly.
DATASET ?= sparklers
RAW_DATA ?= data/raw/$(DATASET).hdf5
WINDOW_ID ?= full
WINDOW_SUFFIX = $(if $(filter full,$(WINDOW_ID)),,/$(WINDOW_ID))
WINDOW_START_US ?=
WINDOW_END_US ?=
WINDOW_MAX_EVENTS ?=
WINDOW_START_INDEX ?=
WINDOW_ARGS = $(if $(WINDOW_START_US),--start-us $(WINDOW_START_US)) \
              $(if $(WINDOW_END_US),--end-us $(WINDOW_END_US)) \
              $(if $(WINDOW_MAX_EVENTS),--max-events $(WINDOW_MAX_EVENTS)) \
              $(if $(WINDOW_START_INDEX),--start-index $(WINDOW_START_INDEX))
METADATA_DIR ?= results/metadata
INSPECT_REPORT ?= $(METADATA_DIR)/$(DATASET).metadata.json
PREPARED_DIR ?= data/processed/$(DATASET)$(WINDOW_SUFFIX)
PREPARED_DATA ?= $(PREPARED_DIR)/events.bin
BINS ?= 5
VOXEL_DIR ?= results/voxels/$(DATASET)/$(WINDOW_ID)
VOXEL_OUTPUT ?= $(VOXEL_DIR)/bins_$(BINS).npy
VALIDATION_DIR ?= results/validation/$(DATASET)/$(WINDOW_ID)
VALIDATION_REPORT ?= $(VALIDATION_DIR)/bins_$(BINS).json
PRECISION_REPORT ?= $(VALIDATION_DIR)/bins_$(BINS).precision.json
VISUAL_OUTPUT ?= results/visualizations/$(DATASET)/$(WINDOW_ID)/bins_$(BINS).png
RPG_REPO ?= external/rpg_e2depth

BENCH_BINS ?= 2,5,10
BENCH_MIN_TIME ?= 2s
BENCH_WARMUP ?= 1
BENCH_REPETITIONS ?= 10
BENCH_FILTER ?= BM_CpuVoxelization
BENCH_OUTPUT_DIR ?= results/benchmarks/$(DATASET)/$(WINDOW_ID)
BENCH_ALLOW_DIRTY ?= 0
BENCH_REQUIRE_VALIDATION ?= 0
SUMMARY_INCLUDE_DIRTY ?= 0
MATRIX_CONFIG ?= configs/benchmark_matrix.json
MATRIX_ALLOW_DIRTY ?= 0

VERIFY_DATASETS := $(basename $(notdir $(wildcard data/raw/*.hdf5)))
VERIFY_BINS ?= 2 5 10
VERIFY_MAX_EVENTS ?= 100000
SUMMARY_WINDOW_ID ?= first_$(VERIFY_MAX_EVENTS)
SYNTH_CASE ?= uniform_100k
SYNTH_EVENTS ?= 100000
SYNTH_WIDTH ?= 640
SYNTH_HEIGHT ?= 480
SYNTH_DURATION_US ?= 100000
SYNTH_SEED ?= 42
SYNTH_SPATIAL ?= uniform
SYNTH_TEMPORAL ?= uniform
SYNTH_HOTSPOT_FRACTION ?= 0.1
SYNTH_HOTSPOT_PROBABILITY ?= 0.9
SYNTH_POSITIVE_PROBABILITY ?= 0.5

.PHONY: help setup setup-external setup-ecf configure build test test-release test-python all
.PHONY: format format-check lint data-inspect data-inspect-all data-prepare synthetic-prepare
.PHONY: cpu-run visualize verify verify-release verify-all verify-coverage validation-summary
.PHONY: benchmark-build benchmark-smoke benchmark benchmark-checked benchmark-summary benchmark-matrix

help:
	@printf '%s\n' \
		'make setup                         Install pinned Python dependencies' \
		'make setup-external                Check out pinned ECF and RPG sources' \
		'make setup-ecf                     Build the pinned HDF5 decoder' \
		'make all                           Build and test CPU Debug' \
		'make test-release                  Build and test CPU Release' \
		'make test-python                   Run pipeline round-trip tests' \
		'make data-inspect-all              Inspect every raw recording' \
		'make data-prepare DATASET=name     Prepare one recording/window' \
		'make verify-all                    Verify a sample from every recording' \
		'make verify-coverage               Verify first, middle, and last samples' \
		'make benchmark-checked             Verify benchmark bins, then measure' \
		'make benchmark-matrix              Run the configured cross-dataset matrix' \
		'make format / make format-check    Format or check C++ source'

.venv/bin/python:
	$(PYTHON_FOR_VENV) -m venv .venv

setup: .venv/bin/python
	$(PIP) install -r requirements.txt

setup-external:
	bash scripts/setup_external.sh all

setup-ecf:
	bash scripts/setup_external.sh ecf
	cmake -S external/hdf5_ecf -B build/hdf5_ecf -G Ninja \
		-DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
	cmake --build build/hdf5_ecf --parallel

configure:
	cmake --preset $(CMAKE_PRESET)

build: configure
	cmake --build --preset $(CMAKE_PRESET) --parallel

test: build
	ctest --preset $(CMAKE_PRESET)

all: test

test-release:
	cmake --preset local-release
	cmake --build --preset local-release --parallel
	ctest --preset local-release

test-python: build
	EVENT_VOXEL_RUN_CPU="$(abspath $(BUILD_DIR)/event_voxel_run_cpu)" \
		$(PYTHON) -m unittest discover -s tests/python -p 'test_*.py' -v

format:
	bash scripts/format.sh

format-check:
	bash scripts/check-format.sh

lint: format-check
	pre-commit run --all-files

data-inspect:
	source scripts/hdf5_env.sh && $(PYTHON) tools/inspect_dataset.py \
		--input "$(RAW_DATA)" --output "$(METADATA_DIR)"

data-inspect-all:
	source scripts/hdf5_env.sh && $(PYTHON) tools/inspect_dataset.py \
		--input data/raw --output "$(METADATA_DIR)"

data-prepare: data-inspect
	@if [[ "$(WINDOW_ID)" == full && -n "$(WINDOW_START_US)$(WINDOW_END_US)$(WINDOW_MAX_EVENTS)$(WINDOW_START_INDEX)" ]]; then \
		echo 'Set WINDOW_ID when selecting a subset of events' >&2; exit 2; fi
	source scripts/hdf5_env.sh && $(PYTHON) tools/prepare_dataset.py \
		--input "$(RAW_DATA)" --report "$(INSPECT_REPORT)" \
		--output "$(PREPARED_DIR)" $(WINDOW_ARGS)

synthetic-prepare:
	$(PYTHON) tools/generate_synthetic.py --output "data/processed/synthetic/$(SYNTH_CASE)" \
		--events $(SYNTH_EVENTS) --width $(SYNTH_WIDTH) --height $(SYNTH_HEIGHT) \
		--duration-us $(SYNTH_DURATION_US) --seed $(SYNTH_SEED) \
		--spatial $(SYNTH_SPATIAL) --temporal $(SYNTH_TEMPORAL) \
		--hotspot-fraction $(SYNTH_HOTSPOT_FRACTION) \
		--hotspot-probability $(SYNTH_HOTSPOT_PROBABILITY) \
		--positive-probability $(SYNTH_POSITIVE_PROBABILITY)

cpu-run: build
	$(BUILD_DIR)/event_voxel_run_cpu "$(PREPARED_DATA)" "$(BINS)" "$(VOXEL_OUTPUT)"

visualize: cpu-run
	$(PYTHON) tools/visualize_voxels.py "$(VOXEL_OUTPUT)" \
		--output "$(VISUAL_OUTPUT)"

verify: cpu-run
	$(PYTHON) tools/verify_external.py --events "$(PREPARED_DATA)" \
		--voxels "$(VOXEL_OUTPUT)" --rpg-repo "$(RPG_REPO)" \
		--report "$(VALIDATION_REPORT)" \
		--cpu-executable "$(BUILD_DIR)/event_voxel_run_cpu"
	$(PYTHON) tools/verify_precision.py --events "$(PREPARED_DATA)" \
		--voxels "$(VOXEL_OUTPUT)" --report "$(PRECISION_REPORT)" \
		--cpu-executable "$(BUILD_DIR)/event_voxel_run_cpu"

benchmark-build:
	cmake --preset benchmark-release
	cmake --build --preset benchmark-release --parallel

verify-release: benchmark-build
	ctest --preset benchmark-release
	$(BENCH_DIR)/event_voxel_run_cpu "$(PREPARED_DATA)" "$(BINS)" "$(VOXEL_OUTPUT)"
	$(PYTHON) tools/verify_external.py --events "$(PREPARED_DATA)" \
		--voxels "$(VOXEL_OUTPUT)" --rpg-repo "$(RPG_REPO)" \
		--report "$(VALIDATION_REPORT)" \
		--cpu-executable "$(BENCH_DIR)/event_voxel_run_cpu"
	$(PYTHON) tools/verify_precision.py --events "$(PREPARED_DATA)" \
		--voxels "$(VOXEL_OUTPUT)" --report "$(PRECISION_REPORT)" \
		--cpu-executable "$(BENCH_DIR)/event_voxel_run_cpu"

# The first N events are a deterministic, memory-bounded sample of each file.
# Reports for all seven recordings remain separate and addressable.
verify-all: benchmark-build
	ctest --preset benchmark-release --output-on-failure
	@for dataset in $(VERIFY_DATASETS); do \
		$(MAKE) data-prepare DATASET="$$dataset" WINDOW_ID=first_$(VERIFY_MAX_EVENTS) \
			WINDOW_MAX_EVENTS=$(VERIFY_MAX_EVENTS) || exit $$?; \
		for bins in $(VERIFY_BINS); do \
			input="data/processed/$$dataset/first_$(VERIFY_MAX_EVENTS)/events.bin"; \
			voxels="results/voxels/$$dataset/first_$(VERIFY_MAX_EVENTS)/bins_$$bins.npy"; \
			report="results/validation/$$dataset/first_$(VERIFY_MAX_EVENTS)/bins_$$bins.json"; \
			precision="results/validation/$$dataset/first_$(VERIFY_MAX_EVENTS)/bins_$$bins.precision.json"; \
			$(BENCH_DIR)/event_voxel_run_cpu "$$input" "$$bins" "$$voxels" || exit $$?; \
			$(PYTHON) tools/verify_external.py --events "$$input" --voxels "$$voxels" \
				--rpg-repo "$(RPG_REPO)" --report "$$report" \
				--cpu-executable "$(BENCH_DIR)/event_voxel_run_cpu" || exit $$?; \
			$(PYTHON) tools/verify_precision.py --events "$$input" --voxels "$$voxels" \
				--report "$$precision" \
				--cpu-executable "$(BENCH_DIR)/event_voxel_run_cpu" || exit $$?; \
		done; \
	done
	$(MAKE) validation-summary VERIFY_MAX_EVENTS=$(VERIFY_MAX_EVENTS) VERIFY_BINS='$(VERIFY_BINS)'

validation-summary:
	$(PYTHON) tools/summarize_validation.py --window-id $(SUMMARY_WINDOW_ID) \
		--bins $(VERIFY_BINS) \
		--output results/reports/validation_$(SUMMARY_WINDOW_ID).json

# Cover distinct event positions as well as the first-N baseline. Source
# inspection remains a full-file check; voxel comparisons use bounded samples.
verify-coverage: verify-all
	@for dataset in $(VERIFY_DATASETS); do \
		count=$$($(PYTHON) -c 'import json,sys; print(json.load(open(sys.argv[1]))["statistics"]["event_count"])' \
			"results/metadata/$$dataset.metadata.json") || exit $$?; \
		if (( count < $(VERIFY_MAX_EVENTS) )); then \
			echo "Recording $$dataset has fewer than $(VERIFY_MAX_EVENTS) events" >&2; exit 2; fi; \
		for position in middle tail; do \
			if [[ "$$position" == middle ]]; then \
				start=$$(( (count - $(VERIFY_MAX_EVENTS)) / 2 )); \
			else start=$$(( count - $(VERIFY_MAX_EVENTS) )); fi; \
			window="$${position}_$(VERIFY_MAX_EVENTS)"; \
			$(MAKE) data-prepare DATASET="$$dataset" WINDOW_ID="$$window" \
				WINDOW_START_INDEX="$$start" WINDOW_MAX_EVENTS=$(VERIFY_MAX_EVENTS) || exit $$?; \
			for bins in $(VERIFY_BINS); do \
				input="data/processed/$$dataset/$$window/events.bin"; \
				voxels="results/voxels/$$dataset/$$window/bins_$$bins.npy"; \
				report="results/validation/$$dataset/$$window/bins_$$bins.json"; \
				precision="results/validation/$$dataset/$$window/bins_$$bins.precision.json"; \
				$(BENCH_DIR)/event_voxel_run_cpu "$$input" "$$bins" "$$voxels" || exit $$?; \
				$(PYTHON) tools/verify_external.py --events "$$input" --voxels "$$voxels" \
					--rpg-repo "$(RPG_REPO)" --report "$$report" \
					--cpu-executable "$(BENCH_DIR)/event_voxel_run_cpu" || exit $$?; \
				$(PYTHON) tools/verify_precision.py --events "$$input" --voxels "$$voxels" \
					--report "$$precision" \
					--cpu-executable "$(BENCH_DIR)/event_voxel_run_cpu" || exit $$?; \
			done; \
		done; \
	done
	$(MAKE) validation-summary SUMMARY_WINDOW_ID=middle_$(VERIFY_MAX_EVENTS) VERIFY_BINS='$(VERIFY_BINS)'
	$(MAKE) validation-summary SUMMARY_WINDOW_ID=tail_$(VERIFY_MAX_EVENTS) VERIFY_BINS='$(VERIFY_BINS)'

benchmark-smoke: benchmark-build
	EVENT_VOXEL_BENCH_INPUT="$(abspath $(PREPARED_DATA))" \
	EVENT_VOXEL_BENCH_BINS="$(BENCH_BINS)" \
		$(BENCH_DIR)/event_voxel_bench_cpu --benchmark_dry_run \
		--benchmark_filter="$(BENCH_FILTER)"

benchmark: benchmark-build
	$(PYTHON) tools/run_benchmark.py --input "$(PREPARED_DATA)" \
		--executable "$(BENCH_DIR)/event_voxel_bench_cpu" \
		--output-dir "$(BENCH_OUTPUT_DIR)" --bins "$(BENCH_BINS)" \
		--filter "$(BENCH_FILTER)" --min-time "$(BENCH_MIN_TIME)" \
		--warmup "$(BENCH_WARMUP)" --repetitions "$(BENCH_REPETITIONS)" \
		$(if $(filter 1,$(BENCH_REQUIRE_VALIDATION)),--validation-dir "$(VALIDATION_DIR)",) \
		$(if $(filter 1,$(BENCH_ALLOW_DIRTY)),--allow-dirty,)

benchmark-checked: benchmark-build
	ctest --preset benchmark-release --output-on-failure
	@IFS=, read -ra bins <<< "$(BENCH_BINS)"; \
	for bin in "$${bins[@]}"; do \
		voxels="$(VOXEL_DIR)/bins_$$bin.npy"; \
		report="$(VALIDATION_DIR)/bins_$$bin.json"; \
		precision="$(VALIDATION_DIR)/bins_$$bin.precision.json"; \
		$(BENCH_DIR)/event_voxel_run_cpu "$(PREPARED_DATA)" "$$bin" "$$voxels" || exit $$?; \
		$(PYTHON) tools/verify_external.py --events "$(PREPARED_DATA)" --voxels "$$voxels" \
			--rpg-repo "$(RPG_REPO)" --report "$$report" \
			--cpu-executable "$(BENCH_DIR)/event_voxel_run_cpu" || exit $$?; \
		$(PYTHON) tools/verify_precision.py --events "$(PREPARED_DATA)" --voxels "$$voxels" \
			--report "$$precision" \
			--cpu-executable "$(BENCH_DIR)/event_voxel_run_cpu" || exit $$?; \
	done
	$(MAKE) benchmark BENCH_REQUIRE_VALIDATION=1

benchmark-summary:
	$(PYTHON) tools/summarize_benchmarks.py --runs "$(BENCH_OUTPUT_DIR)" \
		--output "results/reports/$(DATASET)/$(WINDOW_ID)/cpu_summary.json" \
		$(if $(filter 1,$(SUMMARY_INCLUDE_DIRTY)),--include-dirty,)

benchmark-matrix: benchmark-build
	$(PYTHON) tools/run_benchmark_matrix.py --config "$(MATRIX_CONFIG)" \
		--executable "$(BENCH_DIR)/event_voxel_bench_cpu" \
		$(if $(filter 1,$(MATRIX_ALLOW_DIRTY)),--allow-dirty,)
