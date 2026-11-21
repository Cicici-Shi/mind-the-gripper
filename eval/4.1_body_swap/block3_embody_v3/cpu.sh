#!/bin/bash
# usage: cpu.sh <cmd...>   runs inside embody_v3 with the oft env (CPU node)
export TMPDIR=$HOME/tmp PYTHONFAULTHANDLER=1 MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 ORIGINAL_ONLY=1; unset DISPLAY
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
cd $PROJ_ROOT/embody_v3
exec "$@"
