#!/bin/bash
# next (2026-11-06): copy of vla_zoo/tools/zoo_cells.sh (frozen models1, sha256 3aa7ac57...) -- same cell -> BodyEnv switch
# table, plus: (1) GRP_MAP comes from NEXT_GRP_MAP (default kp20 = models1 decisions 1; the Robotiq proprio mapping for
# the new proprio-reading models is an OPEN decision, see WORKLOG), (2) short aliases S T0 G GTfc GTw GTa G_T.
# EMBODY_DIR is set by the sbatch (default $PROJ_ROOT/embody_toy), not here.
#   cells_next.sh vars <cell>                 -> "BODY VAR=VAL ..." (BODY=SRC for native)
#   cells_next.sh list <suite> <offsets> <cell>...  -> LIST lines "NAME BODY SUITE OFFSETS VAR=VAL..."
set -eu
GM=${NEXT_GRP_MAP:-kp20}
canon() {
  case $1 in S) echo sham_cg;; T0) echo sham_twin;; G) echo grpkp100;; GTfc) echo g_tfc;; GTw) echo g_tw_fc;;
             GTa) echo g_ta_fc;; G_T) echo g_t;; SRC) echo src;; *) echo $1;; esac
}
vars() {
  local c0=${1#obj_}; c0=${c0/#pf_obj_/pf_}; local c=$(canon $c0)
  local G="PandaR85 GRP_KP=100 GRP_MAP=$GM COPY_GRIP_CMD=1"
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
    # physfix (protocol_physfix.md 1-2, Astra rulings 2026-11-06): embody_next only (EMBODY_DIR set by run_oft_pf.sbatch).
    # S / T0 = models1 sham_cg / sham_twin switches; G-prime = R85 kp100 + armature 1e-3 + kp100 proprio map;
    # G-dprime = stock Robotiq kp 20 (GRP_KP unset), kp20 map (physically right for kp20). Twin cells as g_tfc.
    pf_sham_cg)     echo "Panda GRP_MAP=kp20 COPY_GRIP_CMD=1" ;;
    pf_sham_twin)   echo "Panda GRP_MAP=kp20 COPY_GRIP_CMD=1 $TW TWIN_CAMS=both TWIN_CAM_SYNC=1" ;;
    pf_grpkp100a)   echo "PandaR85 GRP_KP=100 GRP_MAP=kp100 R85_ARMATURE=1e-3 COPY_GRIP_CMD=1" ;;
    pf_g_tfc_a)     echo "PandaR85 GRP_KP=100 GRP_MAP=kp100 R85_ARMATURE=1e-3 COPY_GRIP_CMD=1 $TW TWIN_CAMS=both TWIN_CAM=phys TWIN_CAM_SYNC=1" ;;
    pf_g_tloc_a)    echo "PandaR85 GRP_KP=100 GRP_MAP=kp100 R85_ARMATURE=1e-3 COPY_GRIP_CMD=1 $TW TWIN_CAMS=both TWIN_CAM=phys TWIN_CAM_SYNC=1 TOY_PARTS=hand TOY_PHYS_GRIPPER=hide_same TOY_TWIN_POSE=ik" ;;
    pf_grpkp100a_m20) echo "PandaR85 GRP_KP=100 GRP_MAP=kp20 R85_ARMATURE=1e-3 COPY_GRIP_CMD=1" ;;
    pf_g_tfc_a_m20) echo "PandaR85 GRP_KP=100 GRP_MAP=kp20 R85_ARMATURE=1e-3 COPY_GRIP_CMD=1 $TW TWIN_CAMS=both TWIN_CAM=phys TWIN_CAM_SYNC=1" ;;
    pf_grpkp20)     echo "PandaR85 GRP_MAP=kp20 COPY_GRIP_CMD=1" ;;
    pf_g_tfc_kp20)  echo "PandaR85 GRP_MAP=kp20 COPY_GRIP_CMD=1 $TW TWIN_CAMS=both TWIN_CAM=phys TWIN_CAM_SYNC=1" ;;
    *) echo "unknown cell $1" >&2; exit 2 ;;
  esac
}
case $1 in
  vars) vars $2 ;;
  list) suite=$2; offs=$3; shift 3
        for c in "$@"; do n=$(canon $c); set -- $(vars $c); b=$1; shift; echo "$n $b $suite $offs $*"; done ;;
  *) echo "usage: $0 vars <cell> | list <suite> <offsets> <cell>..." >&2; exit 2 ;;
esac
