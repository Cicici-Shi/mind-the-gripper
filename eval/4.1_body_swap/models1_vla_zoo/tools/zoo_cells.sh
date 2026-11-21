#!/bin/bash
# models1 (protocol_models1.md section 2), 2026-11-04: the one place that maps a cell name to its BodyEnv switches.
# Every cell: EMBODY_DIR=$PROJ_ROOT/embody_toy (read-only import), COPY_GRIP_CMD=1, GRP_MAP=kp20 (published mapping,
# protocol_models1_decisions.md 1); every toy-only switch is left unset. Object cells use the same names with obj_.
#   zoo_cells.sh vars <cell>                 -> "BODY VAR=VAL ..." (BODY=SRC for native)
#   zoo_cells.sh list <suite> <offsets> <cell>...  -> run_zoo.sbatch LIST lines "NAME BODY SUITE OFFSETS VAR=VAL..."
set -eu
vars() {
  local c=${1#obj_}
  local G="PandaR85 GRP_KP=100 GRP_MAP=kp20 COPY_GRIP_CMD=1"
  local TW="TWIN_RENDER=1 TWIN_BODY=Panda"
  case $c in
    src)       echo "SRC COPY_GRIP_CMD=1" ;;
    sham_cg)   echo "Panda GRP_MAP=kp20 COPY_GRIP_CMD=1" ;;
    sham_twin) echo "Panda GRP_MAP=kp20 COPY_GRIP_CMD=1 $TW TWIN_CAMS=both TWIN_CAM_SYNC=1" ;;
    grpkp100)  echo "$G" ;;
    g_tfc)     echo "$G $TW TWIN_CAMS=both TWIN_CAM=phys TWIN_CAM_SYNC=1" ;;
    g_tw_fc)   echo "$G $TW TWIN_CAMS=wrist TWIN_CAM=phys TWIN_CAM_SYNC=1" ;;
    g_ta_fc)   echo "$G $TW TWIN_CAMS=agent TWIN_CAM=phys TWIN_CAM_SYNC=1" ;;
    g_t)       echo "$G $TW TWIN_CAMS=both TWIN_CAM=twin TWIN_CAM_SYNC=0" ;;
    *) echo "unknown cell $1" >&2; exit 2 ;;
  esac
}
case $1 in
  vars) vars $2 ;;
  list) suite=$2; offs=$3; shift 3
        for c in "$@"; do set -- $(vars $c); b=$1; shift; echo "$c $b $suite $offs $*"; done ;;
  *) echo "usage: $0 vars <cell> | list <suite> <offsets> <cell>..." >&2; exit 2 ;;
esac
