import argparse
import torch
import os
from model.ActiveExtract import ActiveExtract  
import numpy as np 
import soundfile as sf 
import torch.utils.data as data
import librosa
import python_speech_features
import cv2 as cv
import tqdm
torch.set_printoptions(threshold=torch.inf)
from tools import audioread, audiowrite, cal_SISNR,cal_logpower


MAX_INT16 = np.iinfo(np.int16).max
EPS = np.finfo(float).eps




class dataset(data.Dataset):
    def __init__(self,
                mix_lst_path,
                audio_direc,
                visual_direc,
                mixture_direc,
                batch_size=1,
                partition='test',
                sampling_rate=16000,
                mix_no=2):

        self.minibatch =[]
        self.audio_direc = audio_direc
        self.visual_direc = visual_direc
        self.mixture_direc = mixture_direc
        self.sampling_rate = sampling_rate
        self.partition = partition
        self.C=mix_no
        self.fps = 25
        self.sampling_rate = 16000

        mix_csv=open(mix_lst_path).read().splitlines()
        self.mix_lst=list(filter(lambda x: x.split(',')[0]==partition, mix_csv))

    def __getitem__(self, index):
        line = self.mix_lst[index]
        mixture_path=self.mixture_direc+self.partition+'/mix/'+ line.replace(',','_').replace('/','_')+'.wav'
        mixture,sr = audioread(mixture_path)
        if sr != self.sampling_rate:
            mixture = librosa.resample(mixture,orig_sr=sr, target_sr=self.sampling_rate) 
        #generate mfcc of mix_audio 
        # fps is not always 25, in order to align the visual, we modify the window and step in MFCC extraction process based on fps
        mix_mfcc = python_speech_features.mfcc((mixture*MAX_INT16).astype(np.int16), self.sampling_rate, numcep = 13, winlen = 0.025 * 25 / self.fps, winstep = 0.010 * 25 / self.fps)

        c = 0

        # read target audio
        audio_path =self.mixture_direc+self.partition+'/s%d/'%(c+1)+ line.replace(',','_').replace('/','_')+'.wav'

        audio,sr = audioread(audio_path)
        if sr != self.sampling_rate:
            audio = librosa.resample(audio,orig_sr=sr, target_sr=self.sampling_rate) 

        audio_mfcc = python_speech_features.mfcc((audio*MAX_INT16).astype(np.int16), self.sampling_rate, numcep = 13, winlen = 0.025 * 25 / self.fps, winstep = 0.010 * 25 / self.fps)

        #read video 
        visual_path=self.visual_direc+ self.partition+'/'+ line.split(',')[1+c*6]+'/'+line.split(',')[2+c*6]+'/'+line.split(',')[3+c*6]+'.npy'
        visual = np.load(visual_path)
        start = int(line.split(',')[c*6+4])
        end = int(line.split(',')[c*6+5])
        visual_npy = visual[round(start/self.sampling_rate*self.fps):round(end/self.sampling_rate*self.fps)]


        roiSequence = []
        roiSize = 112

        for i in range(visual_npy.shape[0]):
            frame = visual_npy[i]
            grayed = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            grayed = cv.resize(grayed, (roiSize*2,roiSize*2))
            roi = grayed[int(roiSize-(roiSize/2)):int(roiSize+(roiSize/2)), int(roiSize-(roiSize/2)):int(roiSize+(roiSize/2))]
            roiSequence.append(roi)
        
        visual = np.asarray(roiSequence)

        # read tgt label info 
        audio_time = line.split(',')[c*6+3].split('_')[-4:]
        audio_time = list(map(int, audio_time))
        label = np.zeros(audio_time[1] - audio_time[0])
        label[audio_time[2] - audio_time[0]: audio_time[3] - audio_time[0]] = 1
        label = label[start:end]

        # read label of speaker activity binary mask
        c = 1
        int_start = int(line.split(',')[c*6+4])
        int_end = int(line.split(',')[c*6+5])
        int_time = line.split(',')[c*6+3].split('_')[-4:]
        int_time = list(map(int, int_time))
        label_int = np.zeros(int_time[1] - int_time[0])
        label_int[int_time[2] - int_time[0]: int_time[3] - int_time[0]] = 1
        label_int = label_int[int_start:int_end]


        K = 16000//25
        length  = min(len(audio)//K,len(mixture)//K)
        length  = min(visual.shape[0],length)

        mixture = mixture[0:length*K]
        audio = audio[0:length*K]
        visual = visual[0:length,...]

        if mix_mfcc.shape[0]<visual.shape[0]*4:
            mix_mfcc = np.pad(mix_mfcc,((0,visual.shape[0]*4-mix_mfcc.shape[0]),(0,0)))
        else:
            mix_mfcc = mix_mfcc[0:visual.shape[0]*4]
        
        label = label[0:K*length]
        label_int = label_int[0:K*length]

        return mixture,mix_mfcc,audio,visual,label,label_int,mixture_path

    def __len__(self):
        return len(self.mix_lst)


def segment_utt(a_mix, a_tgt, a_est, label_tgt, label_int):
    utt_list = []
    for utt in [a_mix, a_tgt, a_est]:
        utt_1 = utt[((label_tgt == label_int) & (label_tgt == 0))] #QQ
        utt_3 = utt[((label_tgt == label_int) & (label_tgt == 1))] #SS
        utt_2 = utt[((label_tgt != label_int) & (label_tgt == 1))] #SQ
        utt_4 = utt[((label_tgt != label_int) & (label_tgt == 0))] #QS
        assert utt.shape[-1] == (utt_1.shape[-1]+ utt_2.shape[-1]+ utt_3.shape[-1]+ utt_4.shape[-1])
        utt_list.append([utt_1,utt_2,utt_3,utt_4])  #qq #sq #ss #qs
    return utt_list[0], utt_list[1], utt_list[2]

def eval_segment_weighted_sisdr(a_mix_u, a_tgt_u,a_est_u, label_tgt, label_int,path):
    utt_list = []
    for utt in [a_mix_u, a_tgt_u,a_est_u]:
        utt_1 = utt[((label_tgt == label_int) & (label_tgt == 0))] # QQ
        utt_2 = utt[((label_tgt != label_int) & (label_tgt == 1))] # SQ
        utt_3 = utt[((label_tgt == label_int) & (label_tgt == 1))] # SS
        utt_4 = utt[((label_tgt != label_int) & (label_tgt == 0))] # QS
        assert utt.shape[-1] == (utt_1.shape[-1]+ utt_2.shape[-1]+ utt_3.shape[-1]+ utt_4.shape[-1])
        utt_list.append([utt_1,utt_2,utt_3,utt_4]) 

    a_mix, a_tgt,a_est = utt_list[0], utt_list[1], utt_list[2]
    a_mix_1, a_mix_2, a_mix_3, a_mix_4= a_mix[0], a_mix[1], a_mix[2], a_mix[3]
    a_tgt_1, a_tgt_2, a_tgt_3, a_tgt_4= a_tgt[0], a_tgt[1], a_tgt[2], a_tgt[3]
    a_est_1, a_est_2, a_est_3, a_est_4= a_est[0], a_est[1], a_est[2], a_est[3]

    sisnr_1, sisnr_2, sisnr_3, sisnr_4, sisnr_5, sisnr_6, sisnr_7 = None, None, None, None, None, None, None

    avg_sisnri = cal_SISNR(a_tgt_u, a_est_u)
    non_ca_energy = cal_logpower(a_est_u)
    
    if a_mix_3.shape[-1]==0:
        if a_mix_2.shape[-1]==0:
            sisnr_1 = cal_logpower(a_est_u) #engery 
            non_ca_energy = None
            pass
        elif a_mix_2.shape[-1]!=0:
            sisnr_2 = avg_sisnri #0% overlap
    else:
        a = a_mix_3.shape[-1]
        b = a_mix_2.shape[-1]+a_mix_3.shape[-1]+a_mix_4.shape[-1]
        ratio = a/b
        if ratio <=0.2: 
            sisnr_3 = avg_sisnri
        elif ratio <=0.4:
            sisnr_4 = avg_sisnri
        elif ratio <=0.6:
            sisnr_5 = avg_sisnri
        elif ratio <=0.8:
            sisnr_6 = avg_sisnri
        elif ratio <=1.0:
            sisnr_7 = avg_sisnri

    return non_ca_energy, sisnr_1, sisnr_2, sisnr_3, sisnr_4, sisnr_5, sisnr_6, sisnr_7

def eval_segment_utt(a_mix, a_tgt,a_est):
    a_mix_1, a_mix_2, a_mix_3, a_mix_4= a_mix[0], a_mix[1], a_mix[2], a_mix[3]
    a_tgt_1, a_tgt_2, a_tgt_3, a_tgt_4= a_tgt[0], a_tgt[1], a_tgt[2], a_tgt[3]
    a_est_1, a_est_2, a_est_3, a_est_4= a_est[0], a_est[1], a_est[2], a_est[3]

    energy_1, sisnr_2, sisnr_3, energy_4 = None, None, None, None

    if a_mix_1.shape[-1]!=0:
        energy_1 = cal_logpower(a_est_1)

    if a_mix_2.shape[-1]!=0:
        sisnr_2 =  cal_SISNR(a_tgt_2, a_est_2)

    if a_mix_3.shape[-1]!=0:
        sisnr_3 =  cal_SISNR(a_tgt_3, a_est_3)

    if a_mix_4.shape[-1]!=0:
        energy_4 = cal_logpower(a_est_4)

    return energy_1, sisnr_2, sisnr_3, energy_4 

def main(args):
    # Model

    model = ActiveExtract()
    model = model.cuda()
    pretrained_model = torch.load(args.checkpoint)['model']
    state = model.state_dict()

    for key in state.keys():
        pretrain_key =  'module.'+key
        if pretrain_key in pretrained_model.keys():
            state[key] = pretrained_model[pretrain_key]
        else:
            print("not %s loaded" % pretrain_key)
    model.load_state_dict(state)

    model.cuda()

    datasets = dataset(
                mix_lst_path=args.mix_lst_path,
                audio_direc=args.audio_direc,
                visual_direc=args.visual_direc,
                mixture_direc=args.mixture_direc,
                mix_no=2)

    test_generator = data.DataLoader(datasets,
            batch_size = 1,
            shuffle = False,
            num_workers = args.num_workers)


    model.eval()
    with torch.no_grad():


        avg_u_sisnr_0 = []
        avg_u_sisnr_1 = []
        avg_u_sisnr_2 = []
        avg_u_sisnr_3 = []
        avg_u_sisnr_4 = []
        avg_u_sisnr_5 = []
        avg_u_sisnr_6 = []
        avg_u_sisnr_7 = []
        avg_total = []


        avg_qq_engery = []
        avg_qs_engery = []
        avg_ss_sisnr = []
        avg_sq_sisnr = []

        print('dataset:',len(test_generator))
        for i, (a_mix,a_mix_mfcc,a_tgt,v_tgt,tgt_label,int_label,path) in tqdm.tqdm(enumerate(test_generator)):
            a_mix = a_mix.cuda().float()
            a_tgt = a_tgt.cuda().float()
            v_tgt = v_tgt.cuda().float()
            a_mix_mfcc = a_mix_mfcc.cuda().float()
            tgt_label = tgt_label.squeeze()
            int_label = int_label.squeeze()

        
            est_a_tgt = model(a_mix,a_mix_mfcc,v_tgt)


            u_sisnr_0, u_sisnr_1, u_sisnr_2, u_sisnr_3, u_sisnr_4, u_sisnr_5, u_sisnr_6, u_sisnr_7 = \
                 eval_segment_weighted_sisdr(a_mix.squeeze(), a_tgt.squeeze(),est_a_tgt.squeeze(), tgt_label, int_label,path)


            if u_sisnr_0!=None:
                avg_u_sisnr_0.append(u_sisnr_0)
            if u_sisnr_1!=None:
                avg_u_sisnr_1.append(u_sisnr_1)
            if u_sisnr_2!=None:
                avg_total.append(u_sisnr_2)
                avg_u_sisnr_2.append(u_sisnr_2)
            if u_sisnr_3!=None:
                avg_total.append(u_sisnr_3)
                avg_u_sisnr_3.append(u_sisnr_3)
            if u_sisnr_4!=None:
                avg_total.append(u_sisnr_4)
                avg_u_sisnr_4.append(u_sisnr_4)
            if u_sisnr_5!=None:
                avg_total.append(u_sisnr_5)
                avg_u_sisnr_5.append(u_sisnr_5)
            if u_sisnr_6!=None:
                avg_total.append(u_sisnr_6)
                avg_u_sisnr_6.append(u_sisnr_6)
            if u_sisnr_7!=None:
                avg_total.append(u_sisnr_7)
                avg_u_sisnr_7.append(u_sisnr_7)

            a_mix_utt, a_tgt_utt, a_est_utt =  segment_utt(a_mix.squeeze(), a_tgt.squeeze(),est_a_tgt.squeeze(),tgt_label,int_label)
            energy_1, sisnr_2, sisnr_3, energy_4  = eval_segment_utt(a_mix_utt, a_tgt_utt,a_est_utt)
            if energy_1 !=None:
                avg_qq_engery.append(energy_1)
            if sisnr_2!=None:
                avg_sq_sisnr.append(sisnr_2)
            if sisnr_3 !=None:
                avg_ss_sisnr.append(sisnr_3)
            if energy_4!=None:
                avg_qs_engery.append(energy_4)



            if args.save:
                est_a_tgt = est_a_tgt.squeeze().cpu().numpy()
                a_tgt = a_tgt.squeeze().cpu().numpy()
                a_mix = a_mix.squeeze().cpu().numpy()
                if not os.path.exists(args.save_dir):
                    os.makedirs(args.save_dir)

                if u_sisnr_1!=None:
                    target_dir = 'TA'
                    name = u_sisnr_1
                if u_sisnr_2!=None:
                    target_dir = 'overlap_0'
                    name = u_sisnr_2
                if u_sisnr_3!=None:
                    target_dir = 'overlap_0_20'
                    name = u_sisnr_3
                if u_sisnr_4!=None:
                    target_dir = 'overlap_20_40'
                    name = u_sisnr_4
                if u_sisnr_5!=None:
                    target_dir = 'overlap_40_60'
                    name = u_sisnr_5
                if u_sisnr_6!=None:
                    target_dir = 'overlap_60_80'
                    name = u_sisnr_6
                if u_sisnr_7!=None:
                    target_dir = 'overlap_80_100'
                    name = u_sisnr_7
                    
                
                if not os.path.exists(str(args.save_dir)+'/'+str(target_dir)+'/'):
                    os.makedirs(str(args.save_dir)+'/'+str(target_dir)+'/')
                
                audiowrite(str(args.save_dir)+'/'+str(target_dir)+'/'+'s_%d_mix.wav'%i,a_mix)
                audiowrite(str(args.save_dir)+'/'+str(target_dir)+'/'+'s_%d_tgt.wav'%i,a_tgt)
                audiowrite(str(args.save_dir)+'/'+str(target_dir)+'/'+'s_%d_est_%.2f.wav'%(i,name),est_a_tgt)


        avg_u_sisnr_0 = sum(avg_u_sisnr_0)/len(avg_u_sisnr_0)
        avg_u_sisnr_1 = sum(avg_u_sisnr_1)/len(avg_u_sisnr_1)
        avg_u_sisnr_2 = sum(avg_u_sisnr_2)/len(avg_u_sisnr_2)
        avg_u_sisnr_3 = sum(avg_u_sisnr_3)/len(avg_u_sisnr_3)
        avg_u_sisnr_4 = sum(avg_u_sisnr_4)/len(avg_u_sisnr_4)
        avg_u_sisnr_5 = sum(avg_u_sisnr_5)/len(avg_u_sisnr_5)
        avg_u_sisnr_6 = sum(avg_u_sisnr_6)/len(avg_u_sisnr_6)
        avg_u_sisnr_7 = sum(avg_u_sisnr_7)/len(avg_u_sisnr_7)
        avg_total_value = sum(avg_total)/len(avg_total)

        print('engery:',avg_u_sisnr_1)
        print('ovalap_0:',avg_u_sisnr_2)
        print('overlap_0_20:',avg_u_sisnr_3)
        print('overlap_20_40:',avg_u_sisnr_4)
        print('overlap_40_60:',avg_u_sisnr_5)
        print('overlap_60_80:',avg_u_sisnr_6)
        print('overlap_80_100:',avg_u_sisnr_7)
        print('avg-SI_SNR:',avg_total_value)
        print('total_',len(avg_total))
        print('*')


        avg_qq_engery = sum(avg_qq_engery)/len(avg_qq_engery)
        avg_sq_sisnr = sum(avg_sq_sisnr)/len(avg_sq_sisnr)
        avg_ss_sisnr = sum(avg_ss_sisnr)/len(avg_ss_sisnr)
        avg_qs_engery = sum(avg_qs_engery)/len(avg_qs_engery)
        print('QQ_engery:',avg_qq_engery)
        print('QS_engery:',avg_qs_engery)
        print('SQ_SI-SNR:',avg_sq_sisnr)
        print('SS_SI-SNR:',avg_ss_sisnr)

        print('*')



if __name__ == '__main__':
    parser = argparse.ArgumentParser("avConv-tasnet")
    
    # Dataloader
    parser.add_argument('--mix_lst_path', type=str, default='/workspace2/junjie/USEV/data/iemocap/new_mixture_data_list_2mix.csv',
                        help='directory including train data')
    parser.add_argument('--audio_direc', type=str, default='/workspace2/junjie/dataset/IEMOCAP/uss/clean/',
                        help='directory including validation data')
    parser.add_argument('--visual_direc', type=str, default='/workspace2/junjie/dataset/IEMOCAP/uss/face_npy/',
                        help='directory including test data')
    parser.add_argument('--mixture_direc', type=str, default='/workspace2/junjie/dataset/IEMOCAP/uss/new_mixture/',
                        help='directory of audio')
    parser.add_argument('--checkpoint', type=str, default='../../Checkpoint/ActiveExtract_IEMOCAP-2mix.pt',
                        help='the path of trained model')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers to generate minibatch')

    parser.add_argument('--save', default=0, type=int,
                        help='whether to save audio')
    parser.add_argument('--save_dir', default='./ActiveExtract/', type=str,
                        help='audio_save_path')



    args = parser.parse_args()

    main(args)