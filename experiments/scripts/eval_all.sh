#!/bin/bash

# Steps array
steps=()

# Generate the steps array
for i in {1..5}; do
	    steps+=($(($i * 5000)))
    done

    # Loop over the array
    for step in "${steps[@]}"
    do
	        # Substitute the step into the command
		    LAUNCH_FLAGS="--dataset_ bridgedata --name test_${step} --split_strategy_ task  \
			        --resume_path_ gs://rail-tpus-andre/logs/jaxrl_m_bridgedata/bridge_muse_do_20230502_011720  --lang_encoder_ muse \
				    --resume_step_ ${step} --eval_checkpoint --dropout_rate_ 0.3" sh experiments/andre/scripts/launch_contrastive.sh contrastive_gpu
		    done

# # Generate the steps array
# for i in {1..19}; do
# 	    step=$(($i * 5000))

# 	        # Substitute the step into the command
# 		    LAUNCH_FLAGS="--dataset_ bridgedata --name fs_${step} --split_strategy_ task  \
# 			        --resume_path_ gs://rail-tpus-andre/logs/jaxrl_m_bridgedata/new_labels_task_20230429_072500  \
# 				    --resume_step_ ${step} --eval_checkpoint" bash experiments/andre/scripts/launch_contrastive.sh contrastive_gpu
# 		    done

