# ActiveExtract 
A pytorch implement of Audio-Visual Active Speaker Extraction for Sparsely Overlapped Multi-talker Speech. 

# Usage  
There are three stages to train ActiveExtract  
1. Pretrain an ASD module using TalkSet. 
> You can train it by yourself according to https://github.com/TaoRuijie/TalkNet-ASD/tree/main or just load it from a pretrained model (Checkpoint/TalkNet_TalkSet.model). 

2. Pretrain ActiveExtract on highly overlapped speech dataset VoxCeleb2-2Mix.  
> The ASD module is fixed during this stage

3. Finetune ActiveExtract on sparsely overlapped speech dataset IEMOCAP-2Mix. 
> The ASD module is fixed during this stage 


You can find trained models in 'Checkpoint' folder.  

Contact Email: mrjunjieli@tju.edu.cn 
