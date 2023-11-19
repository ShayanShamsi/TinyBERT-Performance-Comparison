#!/bin/bash

input_file="params.txt"

python_script="run.py"

# Iterate over each line in the input file
while IFS=, read -r training_type training_method model_name_or_path task num_epochs lr batch_size p_length per_sample_max_grad_norm; do
    python "$python_script" \
        --training_type "$training_type" \
        --training_method "$training_method" \
        --model_name_or_path "$model_name_or_path" \
        --task "$task" \
        --num_epochs "$num_epochs" \
        --lr "$lr" \
        --batch_size "$batch_size" \
        --p_length "$p_length" \
        --per_sample_max_grad_norm "$per_sample_max_grad_norm"
done < "$input_file"
