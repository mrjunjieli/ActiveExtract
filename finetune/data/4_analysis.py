'''
compute overlap ratio 
'''

import os 
import csv 
import numpy as np


mix_lst_path='./new_mixture_data_list_2mix.csv'


TrainTAclips=0 
TrainOverlap_0=0
TrainOverlap_0_20=0
TrainOverlap_20_40=0
TrainOverlap_40_60=0
TrainOverlap_60_80=0
TrainOverlap_80_100=0
Traintotal=0

Train_QQ=0
Train_SQ=0
Train_QS=0
Train_SS=0


ValTAclips=0 
ValOverlap_0=0
ValOverlap_0_20=0
ValOverlap_20_40=0
ValOverlap_40_60=0
ValOverlap_60_80=0
ValOverlap_80_100=0
Valtotal=0

Val_QQ=0
Val_SQ=0
Val_QS=0
Val_SS=0

TestTAclips=0 
TestOverlap_0=0
TestOverlap_0_20=0
TestOverlap_20_40=0
TestOverlap_40_60=0
TestOverlap_60_80=0
TestOverlap_80_100=0
Testtotal=0

Test_QQ=0
Test_SQ=0
Test_QS=0
Test_SS=0

with open(mix_lst_path,'r') as file:
    csvreader  = csv.reader(file)

    for line_list in csvreader:
        
        train_type = line_list[0]
        if train_type=='train':
            
            #read s1
            audio_time = line_list[3].split('_')[-4:]
            audio_time = list(map(int, audio_time))
            label = np.zeros(audio_time[1] - audio_time[0])
            label[audio_time[2] - audio_time[0]: audio_time[3] - audio_time[0]] = 1
            start = int(line_list[4])
            end = int(line_list[5])
            tgt_label = label[start:end]

            # read s2
            audio_time = line_list[6+3].split('_')[-4:]
            audio_time = list(map(int, audio_time))
            label = np.zeros(audio_time[1] - audio_time[0])
            label[audio_time[2] - audio_time[0]: audio_time[3] - audio_time[0]] = 1
            start = int(line_list[6+4])
            end = int(line_list[6+5])
            int_label = label[start:end]

            utt = np.ones(len(tgt_label))
            utt_1 = utt[((tgt_label == int_label) & (tgt_label == 0))] # QQ
            utt_2 = utt[((tgt_label != int_label) & (tgt_label == 1))] # SQ
            utt_3 = utt[((tgt_label == int_label) & (tgt_label == 1))] # SS
            utt_4 = utt[((tgt_label != int_label) & (tgt_label == 0))] # QS
            assert utt.shape[-1] == (utt_1.shape[-1]+ utt_2.shape[-1]+ utt_3.shape[-1]+ utt_4.shape[-1])

            # we select s1 spk as tgt spk here 
            if np.sum(utt_3)==0:
                if np.sum(utt_2)==0:
                    TrainTAclips+=1
                else:
                    TrainOverlap_0+=1
            else:
                a = np.sum(utt_3)
                b = np.sum(utt_2)+np.sum(utt_3)+np.sum(utt_4)
                ratio = a/b
                if ratio <=0.2: 
                    TrainOverlap_0_20 +=1
                elif ratio <=0.4:
                    TrainOverlap_20_40+=1
                elif ratio <=0.6:
                    TrainOverlap_40_60+=1
                elif ratio <=0.8:
                    TrainOverlap_60_80+=1
                elif ratio <=1.0:
                    TrainOverlap_80_100+=1
            Traintotal+=1

            
            Train_SS+= np.sum(utt_3)/16000
            Train_QQ+= np.sum(utt_1)/16000 
            Train_QS +=np.sum(utt_4)/16000
            Train_SQ+=np.sum(utt_2)/16000



        if train_type=='val':

            #read s1
            audio_time = line_list[3].split('_')[-4:]
            audio_time = list(map(int, audio_time))
            label = np.zeros(audio_time[1] - audio_time[0])
            label[audio_time[2] - audio_time[0]: audio_time[3] - audio_time[0]] = 1
            start = int(line_list[4])
            end = int(line_list[5])
            tgt_label = label[start:end]

            # read s2
            audio_time = line_list[6+3].split('_')[-4:]
            audio_time = list(map(int, audio_time))
            label = np.zeros(audio_time[1] - audio_time[0])
            label[audio_time[2] - audio_time[0]: audio_time[3] - audio_time[0]] = 1
            start = int(line_list[6+4])
            end = int(line_list[6+5])
            int_label = label[start:end]

            utt = np.ones(len(tgt_label))
            utt_1 = utt[((tgt_label == int_label) & (tgt_label == 0))] # QQ
            utt_2 = utt[((tgt_label != int_label) & (tgt_label == 1))] # SQ
            utt_3 = utt[((tgt_label == int_label) & (tgt_label == 1))] # SS
            utt_4 = utt[((tgt_label != int_label) & (tgt_label == 0))] # QS
            assert utt.shape[-1] == (utt_1.shape[-1]+ utt_2.shape[-1]+ utt_3.shape[-1]+ utt_4.shape[-1])

            # we select s1 spk as tgt spk here 
            if np.sum(utt_3)==0:
                if np.sum(utt_2)==0:
                    ValTAclips+=1
                else:
                    ValOverlap_0+=1
            else:
                a = np.sum(utt_3)
                b = np.sum(utt_2)+np.sum(utt_3)+np.sum(utt_4)
                ratio = a/b
                if ratio <=0.2: 
                    ValOverlap_0_20 +=1
                elif ratio <=0.4:
                    ValOverlap_20_40+=1
                elif ratio <=0.6:
                    ValOverlap_40_60+=1
                elif ratio <=0.8:
                    ValOverlap_60_80+=1
                elif ratio <=1.0:
                    ValOverlap_80_100+=1
            Valtotal+=1

            
            Val_SS+= np.sum(utt_3)/16000
            Val_QQ+= np.sum(utt_1)/16000 
            Val_QS +=np.sum(utt_4)/16000
            Val_SQ+=np.sum(utt_2)/16000




        if train_type=='test':

            #read s1
            audio_time = line_list[3].split('_')[-4:]
            audio_time = list(map(int, audio_time))
            label = np.zeros(audio_time[1] - audio_time[0])
            label[audio_time[2] - audio_time[0]: audio_time[3] - audio_time[0]] = 1
            start = int(line_list[4])
            end = int(line_list[5])
            tgt_label = label[start:end]

            # read s2
            audio_time = line_list[6+3].split('_')[-4:]
            audio_time = list(map(int, audio_time))
            label = np.zeros(audio_time[1] - audio_time[0])
            label[audio_time[2] - audio_time[0]: audio_time[3] - audio_time[0]] = 1
            start = int(line_list[6+4])
            end = int(line_list[6+5])
            int_label = label[start:end]

            utt = np.ones(len(tgt_label))
            utt_1 = utt[((tgt_label == int_label) & (tgt_label == 0))] # QQ
            utt_2 = utt[((tgt_label != int_label) & (tgt_label == 1))] # SQ
            utt_3 = utt[((tgt_label == int_label) & (tgt_label == 1))] # SS
            utt_4 = utt[((tgt_label != int_label) & (tgt_label == 0))] # QS
            assert utt.shape[-1] == (utt_1.shape[-1]+ utt_2.shape[-1]+ utt_3.shape[-1]+ utt_4.shape[-1])

            # we select s1 spk as tgt spk here 
            if np.sum(utt_3)==0:
                if np.sum(utt_2)==0:
                    TestTAclips+=1
                else:
                    TestOverlap_0+=1
            else:
                a = np.sum(utt_3)
                b = np.sum(utt_2)+np.sum(utt_3)+np.sum(utt_4)
                ratio = a/b
                if ratio <=0.2: 
                    TestOverlap_0_20 +=1
                elif ratio <=0.4:
                    TestOverlap_20_40+=1
                elif ratio <=0.6:
                    TestOverlap_40_60+=1
                elif ratio <=0.8:
                    TestOverlap_60_80+=1
                elif ratio <=1.0:
                    TestOverlap_80_100+=1
            Testtotal+=1

            
            Test_SS+= np.sum(utt_3)/16000
            Test_QQ+= np.sum(utt_1)/16000 
            Test_QS +=np.sum(utt_4)/16000
            Test_SQ+=np.sum(utt_2)/16000



    print('Train, Total:',Traintotal,'TA_cips:',TrainTAclips,'Overlap_0:', TrainOverlap_0,'Overlap_0_20:',TrainOverlap_0_20,\
            'Overlap_20_40:',TrainOverlap_20_40,'Overlap_40_60:',TrainOverlap_40_60,'Overlap_60_80:',TrainOverlap_60_80,\
        'Overlap_80_100:',TrainOverlap_80_100)
    print('Val, Total:',Valtotal,'TA_cips:',ValTAclips,'Overlap_0:', ValOverlap_0,'Overlap_0_20:',ValOverlap_0_20,\
            'Overlap_20_40:',ValOverlap_20_40,'Overlap_40_60:',ValOverlap_40_60,'Overlap_60_80:',ValOverlap_60_80,\
        'Overlap_80_100:',ValOverlap_80_100)
    print('Test, Total:',Testtotal,'TA_cips:',TestTAclips,'Overlap_0:', TestOverlap_0,'Overlap_0_20:',TestOverlap_0_20,\
            'Overlap_20_40:',TestOverlap_20_40,'Overlap_40_60:',TestOverlap_40_60,'Overlap_60_80:',TestOverlap_60_80,\
        'Overlap_80_100:',TestOverlap_80_100)
    
    print('Train SS:',Train_SS/3600,'SQ:',Train_SQ/3600,'QS:',Train_QS/3600,'QQ',Train_QQ/3600)
    print('Val SS:',Val_SS/3600,'SQ:',Val_SQ/3600,'QS:',Val_QS/3600,'QQ',Val_QQ/3600)
    print('Test SS:',Test_SS/3600,'SQ:',Test_SQ/3600,'QS:',Test_QS/3600,'QQ',Test_QQ/3600)
