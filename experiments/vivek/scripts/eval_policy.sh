export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
NAME="$1"
CHECK="$2"

CMD="python experiments/vivek/eval_policy.py \
    --num_timesteps 60 \
    --video_save_path /home/robonet \
    --checkpoint_path gs://rail-tpus-$3/jaxrl_m_bridgedata/$NAME/checkpoint_$CHECK \
    --wandb_run_name widowx-gcrl/jaxrl_m_bridgedata/$NAME"

$CMD --goal_eep "0.3 0.0 0.1" --initial_eep "0.3 0.0 0.1"

