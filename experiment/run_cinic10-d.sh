#!/bin/bash

# 注意下面的代码有bug 会全部一次性运行完毕

# Set the desired parameters
iso_VALUES=("0.001" "0.01" "0.1" "0.5" "1.0")
gc_VALUES=("0.01" "0.1" "0.25" "0.5" "0.75")
lap_noise_VALUES=("0.0001" "0.001" "0.01" "0.05" "0.1")

# 最大并行进程数
MAX_PROCESSES=6
current_processes=0

# Function to run Python script and increment the process counter
run_script() {
  python "$@" &
  current_processes=$((current_processes + 1))
}

# non 方法
run_script vfl_cinic10_training.py --save ./model/CINIC10/shell/defence/non > ../output/non.log 2>&1
if [ $current_processes -eq $MAX_PROCESSES ]; then
  wait -n
  current_processes=$((current_processes - 1))
fi

# max_norm 方法
run_script vfl_cinic10_training.py --save ./model/CINIC10/shell/defence/max_norm --max_norm > ../output/max_norm.log 2>&1
if [ $current_processes -eq $MAX_PROCESSES ]; then
  wait -n
  current_processes=$((current_processes - 1))
fi

# signSGD 方法
run_script vfl_cinic10_training.py --save ./model/CINIC10/shell/defence/signSGD --signSGD > ../output/signSGD.log 2>&1
if [ $current_processes -eq $MAX_PROCESSES ]; then
  wait -n
  current_processes=$((current_processes - 1))
fi




for iso_value in "${iso_VALUES[@]}"; do
  run_script vfl_cinic10_training.py --save "./model/CINIC10/shell/defence/iso${iso_value}" --iso --iso_ratio "${iso_value}" > "../output/iso${iso_value}.log" 2>&1
  if [ $current_processes -eq $MAX_PROCESSES ]; then
  wait -n
  current_processes=$((current_processes - 1))
fi
done

for gc_value in "${gc_VALUES[@]}"; do
  run_script vfl_cinic10_training.py --save "./model/CINIC10/shell/defence/gc${gc_value}" --gc --gc_ratio "${gc_value}"  > "../output/gc${gc_value}.log" 2>&1
  if [ $current_processes -eq $MAX_PROCESSES ]; then
  wait -n
  current_processes=$((current_processes - 1))
  fi
done


for lap_noise_value in "${lap_noise_VALUES[@]}"; do
  run_script vfl_cinic10_training.py --save "./model/CINIC10/shell/defence/lap_noise${lap_noise_value}" --lap_noise --lap_noise_ratio "${lap_noise_value}" > "../output/lap_noise${lap_noise_value}.log" 2>&1
  if [ $current_processes -eq $MAX_PROCESSES ]; then
  wait -n
  current_processes=$((current_processes - 1))
  fi
done

# Wait for all background processes to finish
wait