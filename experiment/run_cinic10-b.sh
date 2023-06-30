#!/bin/bash

# Set the desired parameters
eps_VALUES=("0.1" "0.5" "1.0")
poison_VALUES=("1" "2" "4")


#python vfl_cinic10_training.py --save ./model/CINIC10/t/base --eps 0.1 > ../output/test.log 2>&1 &

for eps_value in "${eps_VALUES[@]}"; do
  # Run the Python script with the current combination of parameters
  python vfl_cinic10_training.py --save "./model/CINIC10/shell/eps${eps_value}" --eps "${eps_value}" --poison 4 > "../output/eps${eps_value}.log" 2>&1 &
done

for poison_value in "${poison_VALUES[@]}"; do
  # Run the Python script with the current combination of parameters
  python vfl_cinic10_training.py --save "./model/CINIC10/shell/poison${poison_value}" --poison "${poison_value}" --eps 1.0 > "../output/poison${poison_value}.log" 2>&1 &
done


# Wait for all background processes to finish
wait