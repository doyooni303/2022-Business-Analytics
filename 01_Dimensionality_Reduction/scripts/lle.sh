NUM_SAMPLES=200
k=10
FIG_FOLDER='figures'
TITLE='LLE_with_'$NUM_SAMPLES'samples_'$k'-Neighbors'

python3 lle/LLE.py\
    --num_samples $NUM_SAMPLES\
    --k $k\
    --fig_folder $FIG_FOLDER\
    --title $TITLE\