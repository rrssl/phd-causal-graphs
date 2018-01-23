#!/bin/sh
METHOD=4
NSAMPLES=1200
NPREV=0
DIR=data/20171220/

python gen_training_set.py $METHOD $NSAMPLES
mv S"$METHOD"H-"$NSAMPLES"samples.npy $DIR
python process_training_set.py "$DIR"S"$METHOD"H-"$NSAMPLES"samples.npy $METHOD $NPREV
python train_estimator.py "$DIR"S"$METHOD"H-"$NSAMPLES"samples.npy "$DIR"S"$METHOD"H-"$NSAMPLES"samples-labels.npy
python eval_training.py "$DIR"S"$METHOD"H-"$NSAMPLES"samples-classifier.pkl "$DIR"candidates.pkl
