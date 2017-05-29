#! /bin/sh

#Created on May 29 2017
#@author: Shiran Dudy
#-------
#This shell script's purpose is to train class classifier for obfuscation encoding detection.

# Determine paths for folders
txt2obfuscate=TBD
path2db='db'
path2folds='db/folders'
path2models='models'
path2results='results'
train_seed=7
valid_seed=113

mkdir $path2db
mkdir $path2folds
mkdir $path2models
mkdir $path2results

# generate synthetic data
echo -------------------------------------
echo Generate Synthetic Data for training
echo -------------------------------------
python3 synthesize.py $train_seed $txt2obfuscate > $path2db/train_data
echo --------------------------------------
echo Generate Synthetic Data for validation
echo --------------------------------------
python3 synthesize.py $valid_seed $txt2obfuscate > $path2db/valid_data

# split to five folds
echo ----------------
echo Five Folds Split
echo ----------------
python split_5.py $path2db/train_data $path2folds train
python split_5.py $path2db/valid_data $path2folds valid

# train
echo -----
echo Train
echo -----
python train.py $train_seed $path2folds $path2models

# evaluate
echo --------
echo Evaluate
echo --------
python evaluate.py $path2folds $path2models
