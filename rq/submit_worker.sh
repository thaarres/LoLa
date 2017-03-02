for i in `seq 15`
  do srun --time=1400 --nodes=1 --gres=gpu:1 -C gpu  --partition=normal run_worker.sh &
done
