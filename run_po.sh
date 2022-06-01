export OMP_NUM_THREADS=1
for random_seed in 1 12 123 1234 12345; do
    python transfer.py --horizon 25 --num_threads 25 --temp_path result/transfer_${random_seed}_po --seed ${random_seed} \
        --load_path result/source --num_sources 2 --target False --partition True

    python transfer.py --horizon 25 --num_threads 25 --temp_path result/transfer_${random_seed}_po --seed ${random_seed} \
        --load_path result/source --partition True --num_sources 2 --load_source True \
        --optimizer Adam --rep_num_feature_update 64 --rep_num_adv_update 64 --rep_num_update 50\
        --opt_sampling True --target True
done