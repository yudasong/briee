export OMP_NUM_THREADS=1
for random_seed in 1 12 123 1234 12345; do
	python main.py --horizon $1 --num_threads $2 --temp_path $3 --exp_name s${random_seed} --seed ${random_seed} 
done