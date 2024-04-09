# 2 cores per process
TPU0="export TPU_VISIBLE_DEVICES=0 TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 TPU_HOST_BOUNDS=1,1,1 TPU_MESH_CONTROLLER_ADDRESS=localhost:8476 TPU_MESH_CONTROLLER_PORT=8476"
TPU1="export TPU_VISIBLE_DEVICES=1 TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 TPU_HOST_BOUNDS=1,1,1 TPU_MESH_CONTROLLER_ADDRESS=localhost:8477 TPU_MESH_CONTROLLER_PORT=8477"
TPU2="export TPU_VISIBLE_DEVICES=2 TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 TPU_HOST_BOUNDS=1,1,1 TPU_MESH_CONTROLLER_ADDRESS=localhost:8478 TPU_MESH_CONTROLLER_PORT=8478"
TPU3="export TPU_VISIBLE_DEVICES=3 TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 TPU_HOST_BOUNDS=1,1,1 TPU_MESH_CONTROLLER_ADDRESS=localhost:8479 TPU_MESH_CONTROLLER_PORT=8479"

# 4 cores per process
TPU01="export TPU_VISIBLE_DEVICES=0,1 TPU_CHIPS_PER_HOST_BOUNDS=1,2,1 TPU_HOST_BOUNDS=1,1,1 TPU_MESH_CONTROLLER_ADDRESS=localhost:8476 TPU_MESH_CONTROLLER_PORT=8476"
TPU23="export TPU_VISIBLE_DEVICES=2,3 TPU_CHIPS_PER_HOST_BOUNDS=1,2,1 TPU_HOST_BOUNDS=1,1,1 TPU_MESH_CONTROLLER_ADDRESS=localhost:8478 TPU_MESH_CONTROLLER_PORT=8478"

TPU0123="export TPU_VISIBLE_DEVICES=0,1,2,3 TPU_CHIPS_PER_HOST_BOUNDS=1,4,1 TPU_HOST_BOUNDS=1,1,1 TPU_MESH_CONTROLLER_ADDRESS=localhost:8476 TPU_MESH_CONTROLLER_PORT=8476"

if [[ $# -ne 2 ]] && [[ $# -ne 1 ]]; then
    echo "Usage: bash experiments/scripts/launch_bridge.sh [config] TPU[0123]+" &&
    exit 1
fi

CONFIG="$1"
NAME="all_multimodal_$CONFIG"

case "$2" in
    "TPU0") eval $TPU0 ;;
    "TPU1") eval $TPU1 ;;
    "TPU2") eval $TPU2 ;;
    "TPU3") eval $TPU3 ;;
    "TPU01") eval $TPU01 ;;
    "TPU23") eval $TPU23 ;;
    "TPU0123") eval $TPU0123 ;;
    "") ;;
    *) echo "Invalid TPU argument" && exit 1 ;;
esac

CMD="python experiments/bridgedata_offline_gc.py \
    --config experiments/configs/offline_multimodal_config.py:$CONFIG \
    --bridgedata_config experiments/configs/bridgedata_config.py:all \
    --name $NAME \
    $LAUNCH_FLAGS"

$CMD
