export PYTHONPATH="/home/u202420081000004/Self-Forcing:$PYTHONPATH"
CUDA_VISIBLE_DEVICES=0,1,2,3,7 torchrun --nproc_per_node 5 --master_port 29501 \
scripts/generate_ode_pairs.py \
--output_folder  /hdd/u202420081000004/ode_generate_dataset \
--caption_path 1500.txt