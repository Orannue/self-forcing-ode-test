CUDA_VISIBLE_DEVICES=1,2,5,6 torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=5235 \
  --rdzv_backend=c10d \
  --rdzv_endpoint 127.0.0.1:29501 \
  train.py \
  --config_path configs/ode_trainer_example_config.yaml \
  --logdir /hdd/u202420081000004/ode_train_init