#!/bin/bash
eval "$(conda shell.bash hook)"

## Read input tooth ids to be processed
tooth_ids="$1 $2 $3 $4"
# tooth_ids="1 9 17 25"

conda activate pointnet 

for tooth_id in $tooth_ids; do 
    echo ""
    echo "Training tooth $tooth_id ..."
    echo "" 
    # python train.py --dataset teeth --tooth_id $tooth_id --log_dir logs/log_tooth_"$tooth_id"
    wait 
    echo ""
    echo "Training of tooth $tooth_id is completed!"
    echo "" 
done 

