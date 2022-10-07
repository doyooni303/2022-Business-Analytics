NUM_SAMPLES=100
N_NEIGHBORS=10
FIG_FOLDER='figures'
TITLE='ISOMAP_with_'$NUM_SAMPLES'samples_'$N_NEIGHBORS'-Graph'

python3 isomap/ISOMAP.py\
    --num_samples $NUM_SAMPLES\
    --n_neighbors $N_NEIGHBORS\
    --fig_folder $FIG_FOLDER\
    --title $TITLE\