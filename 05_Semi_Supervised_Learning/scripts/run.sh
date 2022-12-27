MODEL='ResNet34'
SEED=42
DEVICE=1
ALPHA=0.75
LAMBDA_U=0.75
N_VALID=100
T=0.5
EMA_DECAY=0.999
EPOCHS=10
BATCH_SIZE=8
LR=1e-3


for N_LABELED in {250,500,750} # must be <1000
do

EXPERIMENT_DIR='N_LABELED_'$N_LABELED'_Alpha_'$ALPHA'_Lambda_u_'$LAMBDA_U'_T_'$T'_Epochs_'$EPOCHS'_lr_'$LR
NUM_ITER=$N_LABELED
python3 main.py\
    --model $MODEL\
    --experiment_dir $EXPERIMENT_DIR\
    --seed $SEED\
    --device $DEVICE\
    --n_labeled $N_LABELED\
    --n_valid $N_VALID\
    --num_iter $NUM_ITER\
    --alpha $ALPHA\
    --lambda_u $LAMBDA_U\
    --T $T\
    --ema_decay $EMA_DECAY\
    --epochs $EPOCHS\
    --batch_size $BATCH_SIZE\
    --lr $LR\

done