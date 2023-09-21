import argparse
import torch
import os
from model.ActiveExtract import ActiveExtract  
import sys 
import numpy as np 
import soundfile as sf 
import torch.utils.data as data
import librosa
import python_speech_features
import cv2 as cv
import tqdm
torch.set_printoptions(threshold=torch.inf)
from tools import audioread, audiowrite, cal_SISNR

EPS = np.finfo(float).eps
MAX_INT16 = np.iinfo(np.int16).max





class dataset(data.Dataset):
    def __init__(self,
                mix_lst_path,
                visual_direc,
                mixture_direc,
                batch_size=1,
                partition='test',
                sampling_rate=16000,
                mix_no=2):

        self.minibatch =[]
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
        audio_path=self.mixture_direc+self.partition+'/s%d/'%(c+1)+ line.replace(',','_').replace('/','_')+'.wav'
        audio,sr = audioread(audio_path)
        if sr != self.sampling_rate:
            audio = librosa.resample(audio,orig_sr=sr, target_sr=self.sampling_rate) 


        temp=line.split(',')
        visual_path=self.visual_direc+temp[c*4+1]+'/'+temp[c*4+2]+'/'+temp[c*4+3]+'.mp4'
        captureObj = cv.VideoCapture(visual_path)
        roiSequence = []
        roiSize = 112
        while (captureObj.isOpened()):
            ret, frame = captureObj.read()
            if ret == True:
                grayed = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                grayed = cv.resize(grayed, (roiSize*2,roiSize*2))
                roi = grayed[int(roiSize-(roiSize/2)):int(roiSize+(roiSize/2)), int(roiSize-(roiSize/2)):int(roiSize+(roiSize/2))]
                roiSequence.append(roi)
            else:
                break
        captureObj.release()
        visual = np.asarray(roiSequence)

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

        return mixture, mix_mfcc,audio, visual

    def __len__(self):
        return len(self.mix_lst)



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
                visual_direc=args.visual_direc,
                mixture_direc=args.mixture_direc,
                mix_no=2)

    test_generator = data.DataLoader(datasets,
            batch_size = 1,
            shuffle = False,
            num_workers = args.num_workers)



    model.eval()
    with torch.no_grad():
        avg_sisnri = 0
        avg_sisnr=0

        print('dataset:',len(test_generator))
        for i, (a_mix,a_mix_mfcc,a_tgt,v_tgt) in tqdm.tqdm(enumerate(test_generator)):
            a_mix = a_mix.cuda().float()
            a_tgt = a_tgt.cuda().float()
            v_tgt = v_tgt.cuda().float()
            a_mix_mfcc = a_mix_mfcc.cuda().float()
            est_a_tgt = model(a_mix,a_mix_mfcc,v_tgt)
            sisnr_mix = cal_SISNR(a_tgt, a_mix)
            sisnr_est = cal_SISNR(a_tgt, est_a_tgt)
            sisnri = sisnr_est - sisnr_mix
            avg_sisnri += sisnri
            avg_sisnr+=sisnr_est


            if args.save:
                est_a_tgt = est_a_tgt[0].squeeze().cpu().numpy()
                a_tgt = a_tgt[0].squeeze().cpu().numpy()
                a_mix = a_mix[0].squeeze().cpu().numpy()
                if not os.path.exists(args.save_dir):
                    os.makedirs(args.save_dir)
                audiowrite(str(args.save_dir)+'/'+'s_%d_mix.wav'%i,a_mix)
                audiowrite(str(args.save_dir)+'/'+'s_%d_tgt.wav'%i,a_tgt)
                audiowrite(str(args.save_dir)+'/'+'s_%d_est_%.2f.wav'%(i,sisnr_est),est_a_tgt)
            # print(sisnr_est)
        avg_sisnri = avg_sisnri / (i+1)
        avg_sisnr = avg_sisnr/(i+1)
        print('SI-SNR:',avg_sisnr,'SI-SNRi',avg_sisnri)



if __name__ == '__main__':
    parser = argparse.ArgumentParser("avConv-tasnet")
    
    # Dataloader
    parser.add_argument('--mix_lst_path', type=str, default='../data/audio_mixture/mixture_data_list_2mix.csv',
                        help='directory including train data')
    parser.add_argument('--mixture_direc', type=str, default='/workspace2/junjie/dataset/voxceleb2/audio_mixture/2_mix_min_800/',
                        help='directory of audio')
    parser.add_argument('--visual_direc', type=str, default='/workspace2/junjie/dataset/voxceleb2/mp4/',
                        help='directory including test data')
    parser.add_argument('--checkpoint', type=str, default='../../Checkpoint/ActiveExtract_VoxCeleb2-2mix.pt',
                        help='the path of trained model')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers to generate minibatch')

    parser.add_argument('--save', default=0, type=int,
                        help='whether to save audio')
    parser.add_argument('--save_dir', default='./save_audio/', type=str,
                        help='audio_save_path')



    args = parser.parse_args()

    main(args)