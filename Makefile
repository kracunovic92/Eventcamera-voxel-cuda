SHELL := /bin/bash
.DEFAULT_GOAL := help

CMAKE_PRESET ?= local-debug

.PHONY: help configure build test all test-release format format-check lint

help:
	@printf '%s\n' \
		'make test                           Build and test the selected preset' \
		'make test-release                   Build and test local-release' \
		'make format / make format-check     Format or check C++ and CUDA source'

configure:
	cmake --preset $(CMAKE_PRESET)

build: configure
	cmake --build --preset $(CMAKE_PRESET) --parallel

test: build
	ctest --preset $(CMAKE_PRESET) --output-on-failure

all: test

test-release:
	$(MAKE) test CMAKE_PRESET=local-release

format:
	bash scripts/format.sh

format-check:
	bash scripts/check-format.sh

lint: format-check
	pre-commit run --all-files
