# find all configs in configs/
config_file=configs/pool_stvg_16x16_k5l8.yaml
# the dir of the saved weight
weight_dir=/path/to/MMN-STVG/outputs/output
# select weight to evaluate
weight_file=/path/to/MMN-STVG/outputs/output/pool_model_7e.pth
# test batch size
batch_size=32
# set your gpu id
gpus=0,1,2,3,4,5,6,7
# number of gpus
gpun=8
# please modify it with different value (e.g., 127.0.0.2, 29502) when you run multi 2dtan task on the same machine
master_addr=127.0.0.1
master_port=29588

CUDA_VISIBLE_DEVICES=$gpus python -m torch.distributed.launch \
--nproc_per_node=$gpun --master_addr $master_addr --master_port $master_port \
test_net.py --config-file $config_file --ckpt $weight_file OUTPUT_DIR $weight_dir TEST.BATCH_SIZE $batch_size

