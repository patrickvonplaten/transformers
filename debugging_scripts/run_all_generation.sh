#!/usr/bin/env bash

declare -a model_names=("distilgpt2" "gpt2" "xlm-clm-enfr-1024" "openai-gpt" "xlnet-base-cased")
declare -a values=(1.1 1.2 1.3 1.4 1.5)

for model_name in "${model_names[@]}"
do
	echo "Generate repetition penalty comparison for ${model_name}"
	for value in "${values[@]}"
	do
		python run_generation.py --model_name="${model_name}" --rep_pen="${value}"
	done
done
