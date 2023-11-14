#!/usr/bin/env bash

RECIPE_DIR=${RECIPE_DIR:-$PWD}
CI_PROJECT_DIR=${CI_PROJECT_DIR:-/src}
TEST_ARTIFACTS_DIR=${TEST_ARTIFACTS_DIR:-"test-output"}
TCC_TEST_OUTPUT_DIR=${TCC_TEST_OUTPUT_DIR:-"${PWD}/${TEST_ARTIFACTS_DIR}"}

# Make test output directory if it does not exist
[ -d "${TCC_TEST_OUTPUT_DIR}" ] || mkdir -p ${TCC_TEST_OUTPUT_DIR}

python -m pytest
cp junit.xml coverage.xml ${TCC_TEST_OUTPUT_DIR}/
