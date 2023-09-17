#!/bin/sh

gpu_id=0
continue_from=
if [ -z ${continue_from} ]; then
	log_name='ActiveExtract'$(date '+%Y-%m-%d(%H:%M:%S)')
	mkdir -p logs/$log_name
else
	log_name=${continue_from}
fi

CUDA_VISIBLE_DEVICES="$gpu_id" \
python -W ignore \
-m torch.distributed.launch \
--nproc_per_node=1 \
--master_port=3149 \
main.py \
\
--log_name $log_name \
\
--audio_direc '/workspace2/junjie/dataset/voxceleb2/wav/' \
--visual_direc '/workspace2/junjie/dataset/voxceleb2/mp4/' \
--mix_lst_path '../data/audio_mixture/mixture_data_list_2mix.csv' \
--mixture_direc '../data/audio_mixture/' \
--C 2 \
--epochs 200 \
--max_length 4 \
--dynamic 1 \
\
--effec_batch_size 1 \
--accu_grad 0 \
--batch_size 4 \
--num_workers 4 \
--use_tensorboard 1 \
--lr 1e-4  \
>logs/$log_name/console.log 2>&1
# --continue_from ${continue_from} \



