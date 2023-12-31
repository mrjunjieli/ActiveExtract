# ActiveExtract 
A pytorch implement of Audio-Visual Active Speaker Extraction for Sparsely Overlapped Multi-talker Speech. This paper has been submitted to ICASSP2024.  

Arxiv: https://arxiv.org/pdf/2309.08408.pdf

This project aims at real-world speech scenarios where conversations are sparsely overlapped. 

## Usage  
There are three stages to train ActiveExtract  
1. Pretrain an ASD module using TalkSet. 
> You can train it by yourself according to https://github.com/TaoRuijie/TalkNet-ASD or just load it from a pretrained model (Checkpoint/TalkNet_TalkSet.model). 

2. Pretrain ActiveExtract on highly overlapped speech dataset VoxCeleb2-2Mix.  
> The ASD module is fixed during this stage

3. Finetune ActiveExtract on sparsely overlapped speech dataset IEMOCAP-2Mix. 
> The ASD module is fixed during this stage 


You can find trained models in 'Checkpoint' folder.  

## Demo page 
You can find audio samples from this link: https://activeextract.github.io/  

Contact Email: mrjunjieli@tju.edu.cn 
