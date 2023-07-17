LAUNCH_FLAGS='--dataset_ bridgedata --name pt_5k --split_strategy_ task  \
    --resume_path_ gs://rail-tpus-andre/logs/jaxrl_m_bridgedata/new_labels_task_pt_20230429_072651  \
    --resume_step_ 5000 --eval_checkpoint' \
    sh experiments/andre/scripts/launch_contrastive.sh contrastive_tpu  

LAUNCH_FLAGS='--dataset_ bridgedata --name pt_50k --split_strategy_ task  \
    --resume_path_ gs://rail-tpus-andre/logs/jaxrl_m_bridgedata/new_labels_task_pt_20230429_072651  \
    --resume_step_ 50000 --eval_checkpoint' \
    sh experiments/andre/scripts/launch_contrastive.sh contrastive_tpu  

LAUNCH_FLAGS='--dataset_ bridgedata --name pt_95k --split_strategy_ task  \
    --resume_path_ gs://rail-tpus-andre/logs/jaxrl_m_bridgedata/new_labels_task_pt_20230429_072651  \
    --resume_step_ 95000 --eval_checkpoint' \
    sh experiments/andre/scripts/launch_contrastive.sh contrastive_tpu  

LAUNCH_FLAGS='--dataset_ bridgedata --name fs_5k --split_strategy_ task  \
    --resume_path_ gs://rail-tpus-andre/logs/jaxrl_m_bridgedata/new_labels_task_20230429_072500 \
    --resume_step_ 5000 --eval_checkpoint' \
    sh experiments/andre/scripts/launch_contrastive.sh contrastive_tpu  

LAUNCH_FLAGS='--dataset_ bridgedata --name fs_50k --split_strategy_ task  \
    --resume_path_ gs://rail-tpus-andre/logs/jaxrl_m_bridgedata/new_labels_task_20230429_072500 \
    --resume_step_ 50000 --eval_checkpoint' \
    sh experiments/andre/scripts/launch_contrastive.sh contrastive_tpu  

LAUNCH_FLAGS='--dataset_ bridgedata --name fs_95k --split_strategy_ task  \
    --resume_path_ gs://rail-tpus-andre/logs/jaxrl_m_bridgedata/new_labels_task_20230429_072500 \
    --resume_step_ 950000 --eval_checkpoint' \
    sh experiments/andre/scripts/launch_contrastive.sh contrastive_tpu  

LAUNCH_FLAGS='--dataset_ bridgedata --name zs --split_strategy_ task  \
    ---resume_path_ gs://rail-tpus-andre/logs/jaxrl_m_bridgedata/all_v1_task_contrastive_tpu_20230416_084330 \
    --eval_checkpoint' \
    sh experiments/andre/scripts/launch_contrastive.sh contrastive_tpu  


# --resume_path_ gs://rail-tpus-andre/logs/jaxrl_m_bridgedata/all_v1_task_contrastive_tpu_20230416_084330
