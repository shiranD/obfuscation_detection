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
a='one_classifier'
b='ensenble'
c='recurrent'
path2results='results'
train_seed=7
valid_seed=113

mkdir $path2db
mkdir $path2folds
mkdir $path2models
mkdir $path2models/$a
mkdir $path2models/$b
mkdir $path2models/$c

mkdir $path2results

# generate synthetic data
echo -------------------------------------
echo Generate Synthetic Data for training
echo -------------------------------------
python3 synthesize_xor.py $train_seed $txt2obfuscate > $path2db/train_data
echo --------------------------------------
echo Generate Synthetic Data for validation
echo --------------------------------------
python3 synthesize_xor.py $valid_seed $txt2obfuscate > $path2db/valid_data

# split to five folds
echo ----------------
echo Five Folds Split
echo ----------------
python split_5_xor.py $path2db/train_data $path2folds train
python split_5_xor.py $path2db/valid_data $path2folds valid

# train
echo -----
echo Train
echo -----
python one_classifier_train.py $train_seed $path2folds $path2models/$a
python ensemble_train.py $train_seed $path2folds $path2models/$b
python recurrent_train.py $train_seed $path2folds $path2models/$c

# evaluate
echo --------
echo Evaluate
echo --------
python en_evaluate.py $path2folds $path2models/$b
python rec_evaluate.py $path2folds $path2models/$c
