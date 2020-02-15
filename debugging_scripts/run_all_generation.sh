#!/usr/bin/env bash
model_name=${1}
declare -a values=(1.1 1.2 1.3 1.4 1.5)

for i in "${values[@]}"
do
	python run_generation.py --model_name="${model_name}" --rep_pen="${i}"
done
