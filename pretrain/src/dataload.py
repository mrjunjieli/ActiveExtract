import numpy as np
import math
import torch.distributed as dist
import torch
import torch.utils.data as data
import os
import cv2 as cv
import random 
import soundfile as sf 
import librosa
import python_speech_features
from tools import  audioread  

EPS = np.finfo(float).eps
MAX_INT16 = np.iinfo(np.int16).max
np.random.seed(1234)
random.seed(1234)


class dataset(data.Dataset):
    def __init__(self,
                mix_lst_path,
                audio_direc,
                visual_direc,
                mixture_direc,
                batch_size,
                partition='test',
                sampling_rate=16000,
                max_length=4,
                mix_no=2):

        self.minibatch =[]
        self.audio_direc = audio_direc
        self.visual_direc = visual_direc
        self.mixture_direc = mixture_direc
        self.sampling_rate = sampling_rate
        self.partition = partition
        self.max_length = max_length
        self.C=mix_no
        self.fps = 25

        mix_lst=open(mix_lst_path).read().splitlines()
        mix_lst=list(filter(lambda x: x.split(',')[0]==partition, mix_lst))
  
        if self.partition=='train':
            self.batch_size=batch_size
        else:
            self.batch_size=batch_size

        self.spk_path_dict={}
        for line in mix_lst:
            spk = line.split(',')[2]
            if spk not in self.spk_path_dict.keys():
                self.spk_path_dict[spk] = [line]
            else:
                self.spk_path_dict[spk].append(line)
        
        for spk in self.spk_path_dict.keys():
            self.spk_path_dict[spk].sort()

        sorted_mix_lst = sorted(mix_lst, key=lambda data: float(data.split(',')[-1]), reverse=True)
        if self.partition=='train':
            random.shuffle(sorted_mix_lst)
        start = 0
        while True:
            end = min(len(sorted_mix_lst), start + self.batch_size)
            self.minibatch.append(sorted_mix_lst[start:end])
            if end == len(sorted_mix_lst):
                break
            start = end

    def __getitem__(self, index):
        batch_lst = self.minibatch[index]

        min_length = int(float(batch_lst[-1].split(',')[-1])*self.sampling_rate)

        mixtures=[]
        audios=[]
        visuals=[]
        mixfeatures= []
        audiofatures = []

        for line in batch_lst:
            mixture_path=self.mixture_direc+self.partition+'/mix/'+ line.replace(',','_').replace('/','_')+'.wav'
            mixture,sr = audioread(mixture_path)
            if sr != self.sampling_rate:
                mixture = librosa.resample(mixture,orig_sr=sr, target_sr=self.sampling_rate) 

            #generate mfcc of mix_audio 
            # fps is not always 25, in order to align the visual, we modify the window and step in MFCC extraction process based on fps
            mix_mfcc = python_speech_features.mfcc((mixture*MAX_INT16).astype(np.int16), self.sampling_rate, numcep = 13, winlen = 0.025 * 25 / self.fps, winstep = 0.010 * 25 / self.fps)
            if mix_mfcc.shape[0]<self.max_length*self.fps*4:
                mix_mfcc = np.pad(mix_mfcc, ((0, (self.max_length*self.fps*4) -mix_mfcc.shape[0]), (0,0)), 'edge')

            for c in range(self.C):
                # read target audio
                audio_path=self.mixture_direc+self.partition+'/s%d/'%(c+1)+ line.replace(',','_').replace('/','_')+'.wav'
                audio,sr = audioread(audio_path)
                if sr != self.sampling_rate:
                    audio = librosa.resample(audio,orig_sr=sr, target_sr=self.sampling_rate) 

                audio_mfcc = python_speech_features.mfcc((audio*MAX_INT16).astype(np.int16), self.sampling_rate, numcep = 13, winlen = 0.025 * 25 / self.fps, winstep = 0.010 * 25 / self.fps)
                if audio_mfcc.shape[0]<self.max_length*self.fps*4:
                    audio_mfcc = np.pad(audio_mfcc, ((0, (self.max_length*self.fps*4) -audio_mfcc.shape[0]), (0,0)), 'edge')

                #read video 
                temp=line.split(',')
                visual_path=self.visual_direc+temp[c*4+1]+'/'+temp[c*4+2]+'/'+temp[c*4+3]+'.mp4'
                length = math.floor(min_length/self.sampling_rate*self.fps)
                captureObj = cv.VideoCapture(visual_path)
                roiSequence = []
                roiSize = 112
                roiSize_Y = 135
                roiSize_X = 125
                tgt_frame_list=[]
                while (captureObj.isOpened()):
                    ret, frame = captureObj.read()
                    if ret == True:
                        face = cv.resize(frame, (roiSize*2, roiSize*2))\
                            [0:roiSize_Y,int(roiSize - (roiSize_X / 2)):int(roiSize + (roiSize_X / 2))]
                        face = cv.resize(face,(roiSize,roiSize))
                        tgt_frame_list.append(face)
                        grayed = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                        grayed = cv.resize(grayed, (roiSize*2,roiSize*2))
                        roi = grayed[int(roiSize-(roiSize/2)):int(roiSize+(roiSize/2)), int(roiSize-(roiSize/2)):int(roiSize+(roiSize/2))]
                        roiSequence.append(roi)
                    else:
                        break
                captureObj.release()
                visual = np.asarray(roiSequence)
                if visual.shape[0] < length:
                    visual = np.pad(visual, ( (int(length - visual.shape[0]),0), (0,0), (0,0) ), mode = 'edge')

                visuals.append(visual[:self.max_length*self.fps])
                audios.append(audio[:self.max_length*self.sampling_rate])
                audiofatures.append(audio_mfcc[:self.max_length*self.fps*4,:])


            mixtures.append(mixture[:self.max_length*self.sampling_rate])
            mixtures.append(mixture[:self.max_length*self.sampling_rate])
            mixfeatures.append(mix_mfcc[:self.max_length*self.fps*4,:])
            mixfeatures.append(mix_mfcc[:self.max_length*self.fps*4,:])

        np_mixtures = np.asarray(mixtures)
        np_audios = np.asarray(audios)
        np_visuals = np.asarray(visuals)
        np_mix_mfcc = np.asarray(mixfeatures)

        return np_mixtures,np_mix_mfcc,np_audios,np_visuals

    def __len__(self):
        if self.partition=='train':
            return len(self.minibatch)
        else:
            return len(self.minibatch)



class DistributedSampler(data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # indices = torch.randperm(len(self.dataset), generator=g).tolist()
            ind = torch.randperm(int(len(self.dataset)/self.num_replicas), generator=g)*self.num_replicas
            indices = []
            for i in range(self.num_replicas):
                indices = indices + (ind+i).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        # indices = indices[self.rank:self.total_size:self.num_replicas]
        indices = indices[self.rank*self.num_samples:(self.rank+1)*self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

def get_dataloader(args, partition):
    datasets = dataset(
                mix_lst_path=args.mix_lst_path,
                audio_direc=args.audio_direc,
                visual_direc=args.visual_direc,
                mixture_direc=args.mixture_direc,
                batch_size=args.batch_size,
                max_length=args.max_length,
                partition=partition,
                mix_no=args.C,)

    sampler = DistributedSampler(
        datasets,
        num_replicas=args.world_size,
        rank=args.local_rank) if args.distributed else None

    generator = data.DataLoader(datasets,
            batch_size = 1,
            shuffle = (sampler is None),
            num_workers = args.num_workers,
            sampler=sampler,pin_memory=True)

    return sampler, generator

    
if __name__ == '__main__':
    datasets = dataset(
                mix_lst_path='/workspace2/junjie/ASD_SE/data_process/voxceleb2/mixture_data_list_2mix.csv',
                audio_direc='/workspace2/junjie/dataset/voxceleb2/wav/',
                visual_direc='/workspace2/junjie/dataset/voxceleb2/mp4/',
                mixture_direc='/workspace2/junjie/dataset/voxceleb2/audio_mixture/2_mix_min_800/',
                batch_size=2,
                partition='val')
    data_loader = data.DataLoader(datasets,
                batch_size = 1,
                shuffle= True,
                num_workers = 10)
    save_folder='temp'
    for i, (a_mix,a_mix_mfcc,audio,visual) in enumerate(data_loader):
        print(a_mix.shape) # 1,4,96000
        print(a_mix_mfcc.shape) # 1,4,600,13 
        print(audio.shape) # 
        print(visual.shape) # 1,4,150,112,112
        # print(face.shape) #1,4,112,112,3
        # # pass
        for i in range(4):
            a_tgt = audio.squeeze()[i,:].cpu().numpy()
            mix = a_mix.squeeze()[i,:].cpu().numpy()
            # audiowrite('./%d_tgt.wav'%i,a_tgt)
            # audiowrite('./%d_output.wav'%i,mix)
            for d in range(0,visual.shape[2],4):
                cv.imwrite('./%d_%dtest.jpg'%(i,d),visual[0][i][d].cpu().numpy())
                if d>=20:
                    break
        
            
        break