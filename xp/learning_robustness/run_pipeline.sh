#!/bin/sh
# Usage: ./run_pipeline.sh . 4 1000 ndoms 7
DIR=$(readlink -e "$1")
shift
METHOD="$1"
shift
NSAMPLES="$1"
shift
KWARGS="$@"
BASE=S"$METHOD"H-"$NSAMPLES"

python gen_training_set.py $METHOD $NSAMPLES
mv -t $DIR $BASE-training.npy $BASE-test.npy
TRAIN=$DIR/$BASE-training.npy
TEST=$DIR/$BASE-test.npy

python process_training_set.py $TRAIN $METHOD $KWARGS
python process_training_set.py $TEST $METHOD $KWARGS
TRAIN_LABELS=$DIR/$BASE-training-labels.npy
TEST_LABELS=$DIR/$BASE-test-labels.npy

python train_estimator.py $TRAIN $TRAIN_LABELS
ESTIMATOR=$DIR/$BASE-classifier.pkl

python eval_training.py $ESTIMATOR $TEST $TEST_LABELS
