#! /bin/bash
#instanceNames=("O101200")
#instanceNames=("R101200" "C101200")
#instanceNames=("O101200" "R101200" "C101200")
instanceNames=($3)

#variants=("LRP" "HVRP" "VRPTW" "CVRP" "TSP")
#variants=("LRP" "HVRP" "VRPTW" "CVRP")
#variants=("LRP")
variants=($4)

suffix=$1
expPart=$2

for i in ${!instanceNames[@]};
do
  instanceName=${instanceNames[$i]}

  for j in ${!variants[@]};
  do
    variant=${variants[$j]}
    $ATES_SOURCE_DIR/scripts/run_experiments.sh ${expPart} ${suffix}${variant} ${instanceName} 10 10 5 5 ${variant}

  done

done