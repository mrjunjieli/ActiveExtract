This script is used to generate training, validation and test set from Voxceleb2 


# Usage
```
./run.sh 
```

# Data Structure


```
voxceleb2/ #Your folder of VoxCeleb2 
├── mp4
│   ├── test # The original test set contains .mp4 video
│   └── train # The original train set contains .mp4 video
└── wav
    ├── test # The original test set contains .wav video
    ├── train # The original train set contains .wav audio 

audio_mixture/ # (new) The simulated 2 speaker mixture contatins .wav audio
├── mixture_data_list_2mix.csv # (new) The list of the simulated speech mixtures
├── test
│   ├── mix
│   ├── s1
│   └── s2
├── train
│   ├── mix
│   ├── s1
│   └── s2
└── val
    ├── mix
    ├── s1
    └── s2
```