#!/bin/bash 


#path 
direc=/workspace2/junjie/dataset/voxceleb2/ # Your folder of VoxCeleb2 
data_direc_mp4=${direc}mp4/  #video folder of VoxCeleb2
audio_data_direc=${direc}wav/ # Audio folder of VoxCeleb2 
mixture_audio_direc=./audio_mixture/ # Audio mixture saved directory


train_samples=20000 # no. of train mixture samples simulated
val_samples=5000 # no. of validation mixture samples simulated
test_samples=3000 # no. of test mixture samples simulated
C=2 # only 2 here 
mix_db=10 # random db ratio from -10 to 10db
sampling_rate=16000 # audio sampling rate
min_length=4 # minimum length of audio. Audios will be removed if less than min_length


mixture_data_list=${mixture_audio_direc}/mixture_data_list_${C}mix.csv #mixture datalist


# stage 1: Remove repeated datas in pretrain and train set, extract audio from mp4, create mixture list
if [ ! -f $mixture_data_list ];then
    echo $mixture_data_list' not exist'
    echo 'stage 1: create mixture list'
    python 1_create_mixture_list.py \
    --data_direc $data_direc_mp4 \
    --C $C \
    --mix_db $mix_db \
    --train_samples $train_samples \
    --val_samples $val_samples \
    --test_samples $test_samples \
    --audio_data_direc $audio_data_direc \
    --min_length $min_length \  
    --sampling_rate $sampling_rate \
    --mixture_data_list $mixture_data_list 
    else
    echo $mixture_data_list' exist!'
fi


# stage 2: create audio mixture from list
echo 'stage 2: create mixture audios'
python 2_create_mixture.py \
--C $C \
--audio_data_direc $audio_data_direc \
--mixture_audio_direc $mixture_audio_direc \
--mixture_data_list $mixture_data_list \
--sampling_rate $sampling_rate
