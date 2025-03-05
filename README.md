This repository contains the code and data for the article [_A Graph Reinforcement Learning Framework for Neural Adaptive Large Neighbourhood Search_](https://doi.org/10.1016/j.cor.2024.106791) by [Syu-Ning Johnn](https://profiles.ucl.ac.uk/94697-shunee-johnn), [Victor-Alexandru Darvariu](https://victor.darvariu.me), [Julia Handl](https://research.manchester.ac.uk/en/persons/julia.handl) and [Jörg Kalcsics](https://www.maths.ed.ac.uk/school-of-mathematics/people/a-z?person=450), published in Computers & Operations Research.

If you use this code, please consider citing our article:

```biblatex
@article{johnn2024grlalns,
	title = {A Graph Reinforcement Learning framework for neural Adaptive Large Neighbourhood Search},
	journal = {Computers & Operations Research},
	pages = {106791},
	year = {2024},
	doi = {https://doi.org/10.1016/j.cor.2024.106791},
	author = {Johnn, Syu-Ning and Darvariu, Victor-Alexandru and Handl, Julia and Kalcsics, Joerg}, 
}
```

If you run into any issues when using this code, please contact Syu-Ning Johnn at [s.johnn@ucl.ac.uk](s.johnn@ucl.ac.uk) and Victor-Alexandru Darvariu at [victord@robots.ox.ac.uk](mailto:victord@robots.ox.ac.uk).

# License
This code is licensed under the MIT license. Consult the LICENSE file for more details.

# Instructions

## Configuration

After cloning the repository, set the following environment variables e.g. in your `.bashrc`, adjusting paths and directories as needed. The ATES prefix string, used throughout the codebase, is due to the project originating with the [Alan Turing Institute Enrichment Scheme](https://www.turing.ac.uk/opportunities/doctoral-student-opportunities/phd-enrichment-scheme). 

```bash
# Source directories
export ATES_SOURCE_DIR=/home/jane/git/grl-alns-framework
export PATH=$PATH:$ATES_SOURCE_DIR/scripts

# Experiment data and results will be stored here.
export ATES_EXPERIMENT_DATA_DIR=/home/jane/experiments/grl-alns-framework 
```

## Installation

```bash
pip install virtualenv # if needed

# create virtual environment
cd $ATES_SOURCE_DIR && virtualenv venv 
# or, alternatively, with a specific interpreter
cd $ATES_SOURCE_DIR && /usr/bin/python3 -m venv venv 

source venv/bin/activate
pip install --no-cache-dir -r requirements.txt

# install PyTorch. 
# we used Python 3.9 and torch CPU version 1.13.0; the commands below install the appropriate versions.
# for x86 Linux...
pip install --no-cache-dir https://download.pytorch.org/whl/cpu/torch-1.13.0+cpu-cp39-cp39-linux_x86_64.whl
# or for an M1/M2 Mac...
pip install --no-cache-dir https://download.pytorch.org/whl/cpu/torch-1.13.0-cp39-none-macosx_11_0_arm64.whl

# install PyTorch-Geometric (PyG) -- we used version 0.1.0. 
# commands below show how to install the right version from source, which is most generic, but wheels will generally be available for your platform.
pip install --verbose git+https://github.com/pyg-team/pyg-lib.git@2eab973e6c6b5cb1209b0148f4547742a64f4d89
pip install --verbose torch_scatter
pip install --verbose torch_sparse
pip install --verbose torch_cluster
pip install --verbose torch_spline_conv
```

## Datasets
Datasets (i.e., problem instances) that were used for our experiments can be found in the `data` subdirectory. 

The code for reading these files and generating states can be found under `state/state_generator.py`. If you would like to apply the method to your own dataset, you can implement another subclass of `StateGenerator` that defines how the files should be processed.


## Running the algorithm
We make the following scripts available if you would like to try out the algorithms.

To train an operator selector agent, run the following Python script. This runs our GRLOS method by default. You can change the algorithm to one of the baselines and also modify the parameters specified within to suit your needs.

```bash
python $ATES_SOURCE_DIR/experiment_launchers/run_dqn.py
```

To use the trained GRLOS operator selector inside ALNS, run the following command:

```bash
python $ATES_SOURCE_DIR/experiment_launchers/run_alns.py
```

## Reproducing the experiments

We highlight that the paper experiments require a significant amount of compute (*years* of single-core CPU time) and cannot realistically be reproduced quickly on a single desktop machine.

We make several scripts available under `$ATES_SOURCE_DIR/scripts`, and analysis notebooks under `$ATES_SOURCE_DIR/notebooks` for this purpose.

The scripts rely on setting experiment parameters under `$ATES_SOURCE_DIR/evaluation/experiment_conditions.py`. The file currently specifies parameters for small-scale experiments to "smoke-test" that your installation has succeeded. Comments in this file specify values that were used for experiments in the paper.  

## Example of using run_experiments.sh script
```bash
# run the agent training and hyperparameter selection
$ATES_SOURCE_DIR/scripts/run_experiments.sh hyperopt mainCVRP O101200 10 10 5 4 CVRP
# use the trained models for MDP evaluation
$ATES_SOURCE_DIR/scripts/run_experiments.sh eval mainCVRP O101200 10 10 5 4 CVRP
# tune the model for use in ALNS
$ATES_SOURCE_DIR/scripts/run_experiments.sh tune mainCVRP O101200 10 10 5 4 CVRP
# apply inside ALNS loop
$ATES_SOURCE_DIR/scripts/run_experiments.sh alns mainCVRP O101200 10 10 5 4 CVRP
```

You can find the results under `$ATES_EXPERIMENT_DATA_DIR/O101200_mainCVRP` directory:

- `models/eval_results`: evaluation results for the standalone MDP for the different methods;
- `models/alns_results`: evaluation results when integrated in ALNS;
- `models/checkpoints`: best validated checkpoints for the trained models.

You can follow a similar pattern for launching experiments on the other problem variants (TSP/HVRP/VRPTW/LRP) and the C/R instances. You can also check the `scripts/run_all_variants.sh` script. 

## Peeking at running experiments
You might find the following script useful for checking the performance of the models during training for a given experiment: 

```bash
cd $ATES_SOURCE_DIR && source venv/bin/activate && python experiment_launchers/peek_val_losses.py --experiment_id O101200_mainCVRP --parent_dir $ATES_EXPERIMENT_DATA_DIR
```