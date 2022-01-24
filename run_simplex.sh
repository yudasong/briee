export OMP_NUM_THREADS=1
for random_seed in 1 12 123 1234 12345; do
	python main.py --horizon 30 --num_threads $1 --temp_path $2 --variable_latent True --exp_name s${random_seed} --seed ${random_seed} 
done