n_blocks=1 srun --time=1200 --nodes=1 --gres=gpu:1 -C gpu  --partition=normal  run_daint.sh block1 &
sleep 10
n_blocks=3 srun --time=1200 --nodes=1 --gres=gpu:1 -C gpu  --partition=normal  run_daint.sh block3 &
sleep 10
n_conv_layers=1 srun --time=1200 --nodes=1 --gres=gpu:1 -C gpu  --partition=normal  run_daint.sh convlayer1 &
sleep 10
n_conv_layers=3 srun --time=1200 --nodes=1 --gres=gpu:1 -C gpu  --partition=normal  run_daint.sh convlayer3 &
sleep 10
n_conv_layers=1 n_blocks=3 srun --time=1200 --nodes=1 --gres=gpu:1 -C gpu  --partition=normal  run_daint.sh convlayer1 block3 &
sleep 10
conv_size=3 srun --time=1200 --nodes=1 --gres=gpu:1 -C gpu  --partition=normal  run_daint.sh conv3 &
sleep 10
conv_size=5 srun --time=1200 --nodes=1 --gres=gpu:1 -C gpu  --partition=normal  run_daint.sh conv5 &
sleep 10
conv_size=6 srun --time=1200 --nodes=1 --gres=gpu:1 -C gpu  --partition=normal  run_daint.sh conv6 &
sleep 10
n_dense_layers=1 srun --time=1200 --nodes=1 --gres=gpu:1 -C gpu  --partition=normal  run_daint.sh dense1 &
sleep 10
n_dense_layers=3 srun --time=1200 --nodes=1 --gres=gpu:1 -C gpu  --partition=normal  run_daint.sh dense3 &
sleep 10
n_dense_layers=4 srun --time=1200 --nodes=1 --gres=gpu:1 -C gpu  --partition=normal  run_daint.sh dense4 &
sleep 10
n_dense_nodes=128 srun --time=1200 --nodes=1 --gres=gpu:1 -C gpu  --partition=normal  run_daint.sh densenodes128 &
sleep 10
conv_batchnorm=1 srun --time=1200 --nodes=1 --gres=gpu:1 -C gpu  --partition=normal  run_daint.sh convbatchnorm1 &
sleep 10
dense_batchnorm=1 srun --time=1200 --nodes=1 --gres=gpu:1 -C gpu  --partition=normal  run_daint.sh densebatchnorm1 &
sleep 10
pool_size=0 srun --time=1200 --nodes=1 --gres=gpu:1 -C gpu  --partition=normal  run_daint.sh pool0 &
sleep 10
pool_size=4 srun --time=1200 --nodes=1 --gres=gpu:1 -C gpu  --partition=normal  run_daint.sh pool4 &
sleep 10
conv_dropout=0.2 srun --time=1200 --nodes=1 --gres=gpu:1 -C gpu  --partition=normal  run_daint.sh convdropout02 &
sleep 10
block_dropout=0.2 srun --time=1200 --nodes=1 --gres=gpu:1 -C gpu  --partition=normal  run_daint.sh blockdropout02 &
sleep 10
dense_dropout=0.2 srun --time=1200 --nodes=1 --gres=gpu:1 -C gpu  --partition=normal  run_daint.sh densedropout02 &
sleep 10

