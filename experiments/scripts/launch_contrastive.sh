CONFIG=${1-contrastive_tpu}
NAME="all_v1_task_$CONFIG"

CMD="python experiments/andre/contrastive.py \
    --config experiments/andre/configs/offline_contrastive_config.py:$CONFIG \
    --bridgedata_config experiments/andre/configs/bridgedata_config.py:all \
    --name $NAME \
    $LAUNCH_FLAGS"

$CMD
