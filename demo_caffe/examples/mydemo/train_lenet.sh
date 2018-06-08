#!/usr/bin/env sh
set -e

./build/tools/caffe train --solver=examples/mydemo/lenet_solver.prototxt $@
