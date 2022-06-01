export OMP_NUM_THREADS=1
for random_seed in 1 12 123 1234 12345; do
	python transfer.py --horizon 25 --num_threads 25 --temp_path result/transfer_${random_seed}_online --seed ${random_seed} \
        --load_path result/source --num_sources 5 --target True --online True
done