# TOY1 cell definitions (protocol_toy1.md section 2 + protocol_toy1_astra_decisions.md; 2026-11-03).
# usage: source tools/toy_cells.sh; cell_vars toy_g   -> prints "VAR=val ..." for env
cell_vars() {
  local G="BODY=PandaR85 GRP_KP=100 GRP_MAP=kp100 COPY_GRIP_CMD=1"
  case "$1" in
    toy_s)       echo "BODY=Panda COPY_GRIP_CMD=1" ;;
    toy_s_hid)   echo "BODY=Panda COPY_GRIP_CMD=1 HIDE_GRIPPER=all" ;;
    toy_g)       echo "$G" ;;
    toy_g_hid)   echo "$G HIDE_GRIPPER=all" ;;
    toy_g_tfc)   echo "$G TWIN_RENDER=1 TWIN_CAM=phys" ;;
    toy_g_t)     echo "$G TWIN_RENDER=1" ;;
    toy_g_cam)   echo "$G WRIST_CAM_DZ_MM=-6.4" ;;
    toy_g_map20) echo "BODY=PandaR85 GRP_KP=100 GRP_MAP=kp20 COPY_GRIP_CMD=1" ;;
    *) echo "unknown cell $1" >&2; return 1 ;;
  esac
}
TOY_CELLS="toy_s toy_g toy_g_tfc toy_g_hid toy_g_t toy_s_hid toy_g_cam"
