#!/bin/bash
gpu=$CUDA_VISIBLE_DEVICES
IFS=',' read -r -a array <<< "$gpu"
TARGET="No running processes"
for element in "${array[@]}"
do
    status="$(nvidia-smi -i "$element")"
    if echo "$status" | grep -q "$TARGET"; then
        CUDA_VISIBLE_DEVICES="$element"
        echo $CUDA_VISIBLE_DEVICES
        time python main.py --genome_id 1 --nepochs 2
        exit 0
    fi
done

python report_error_evaluation.py --genome_id 1
