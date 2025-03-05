#! /bin/bash
expPart=$1
expSuffix=$2
instanceNames=( $3 )

fixedScalePc=$4
budget=$5
alnsOuterItsPerCust=$6
alnsInnerIts=${7}
problemVariant="${8}"

for i in ${!instanceNames[@]};
do
  instanceName=${instanceNames[$i]}
  echo "Doing instance $instanceName"

  expId=${instanceName}_${expSuffix}
  source $ATES_SOURCE_DIR/venv/bin/activate

  python $ATES_SOURCE_DIR/setup_experiments.py --experiment_id ${expId} --experiment_part ${expPart} \
    --instance_name ${instanceName}  --which main --parent_dir $ATES_EXPERIMENT_DATA_DIR \
    --fixed_scale_pc ${fixedScalePc} \
    --budget ${budget} --alns_outer_its_per_customer ${alnsOuterItsPerCust} --alns_inner_its ${alnsInnerIts} \
    --problem_variant ${problemVariant}

  taskCount=$(cat $ATES_EXPERIMENT_DATA_DIR/${expId}/models/${expPart}_tasks.count | tr -d '\n')

  for i in $(seq 1 $taskCount); do
    (python $ATES_SOURCE_DIR/tasks.py --experiment_id ${expId} --experiment_part ${expPart} --parent_dir $ATES_EXPERIMENT_DATA_DIR --task_id $i > /dev/null) &
  done

done

echo "Done launching everything."
