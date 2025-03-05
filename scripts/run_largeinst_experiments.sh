#! /bin/bash
instanceNames=("O1_10_01" "O1_08_01" "O1_06_01" "O1_04_01" "O1_02_01")
variants=("LRP" "VRPTW" "HVRP" "CVRP" "TSP")
seeds=(0 42 84 126 168 210 252 296 336 378)

for i in ${!instanceNames[@]};
do
  instanceName=${instanceNames[$i]}

  for j in ${!variants[@]};
  do
    variant=${variants[$j]}

      for k in ${!seeds[@]};
      do
        seed=${seeds[$k]}
        python $ATES_SOURCE_DIR/experiment_launchers/rw_largeinst_experiment.py --problem_variant ${variant} --instance_name ${instanceName} --seed ${seed}) & > /dev/null
        sleep 0.25
      done
  done
done