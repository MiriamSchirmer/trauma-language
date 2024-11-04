#!/bin/bash
HYPERFILE="code/hyperparameters.json" # hyperparameter file
export PYTHONPATH="."; python3 train_models.py $HYPERFILE $1 $2 --device $3 # perform search
export PYTHONPATH="."; python3 train_models.py $HYPERFILE $1 $2 --device $3 --retrain true # run best hyperparameters