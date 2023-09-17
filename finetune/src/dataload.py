import numpy as np
import math
import torch.distributed as dist
import torch
import torch.nn as nn
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
                partition='val',
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
        self.batch_size = batch_size


        mix_lst=open(mix_lst_path).read().splitlines()
        mix_lst=list(filter(lambda x: x.split(',')[0]==partition, mix_lst))


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

        min_length = np.inf
        for _ in range(len(batch_lst)):
            if float(batch_lst[_].split(',')[-1])<min_length:
                min_length= float(batch_lst[_].split(',')[-1])
        min_length = math.floor(min_length)
        
        mixtures=[]
        audios=[]
        visuals=[]
        mixfeatures= []
        activate_labels=[]
        audiofeatures=[]


        for line in batch_lst:
            mixture_path=self.mixture_direc+self.partition+'/mix/'+ line.replace(',','_').replace('/','_')+'.wav'
            mixture,sr = audioread(mixture_path)
            if sr != self.sampling_rate:
                mixture = librosa.resample(mixture,orig_sr=sr, target_sr=self.sampling_rate) 
            #generate mfcc of mix_audio 
            # fps is not always 25, in order to align the visual, we modify the window and step in MFCC extraction process based on fps
            mix_mfcc = python_speech_features.mfcc((mixture*MAX_INT16).astype(np.int16), self.sampling_rate, numcep = 13, winlen = 0.025 * 25 / self.fps, winstep = 0.010 * 25 / self.fps)
            
            #truncate 
            mixture=mixture[0:min_length*self.sampling_rate]
            if len(mixture)<min_length*self.sampling_rate:
                mixture = np.pad(mixture,(0,min_length*self.sampling_rate-len(mixture)))

            if mix_mfcc.shape[0]<min_length*self.fps*4:
                mix_mfcc = np.pad(mix_mfcc, ((0, (min_length*self.fps*4) -mix_mfcc.shape[0]), (0,0)), 'edge')
            else:
                mix_mfcc = mix_mfcc[0:min_length*self.fps*4,:]

            for c in range(self.C):
                # read target audio
                audio_path =self.mixture_direc+self.partition+'/s%d/'%(c+1)+ line.replace(',','_').replace('/','_')+'.wav'

                audio,sr = audioread(audio_path)
                audio = audio[0:min_length*self.sampling_rate]
                if sr != self.sampling_rate:
                    audio = librosa.resample(audio,orig_sr=sr, target_sr=self.sampling_rate) 
                if len(audio)<min_length*self.sampling_rate:
                    audio = np.pad(audio,(0,min_length*self.sampling_rate-len(audio)))

                audio_mfcc = python_speech_features.mfcc((audio*MAX_INT16).astype(np.int16), self.sampling_rate, numcep = 13, winlen = 0.025 * 25 / self.fps, winstep = 0.010 * 25 / self.fps)

                #read video 
                visual_path=self.visual_direc+ self.partition+'/'+ line.split(',')[1+c*6]+'/'+line.split(',')[2+c*6]+'/'+line.split(',')[3+c*6]+'.npy'
                visual = np.load(visual_path)
                start = int(line.split(',')[c*6+4])
                end = int(line.split(',')[c*6+5])
                visual_npy = visual[round(start/self.sampling_rate*self.fps):round(end/self.sampling_rate*self.fps)]


                roiSequence = []
                roiSize = 112
                roiSize_Y = 135
                roiSize_X = 125
                tgt_frame_list=[]

                for i in range(visual_npy.shape[0]):
                    frame = visual_npy[i]
                    face = cv.resize(frame, (roiSize*2, roiSize*2))\
                            [0:roiSize_Y,int(roiSize - (roiSize_X / 2)):int(roiSize + (roiSize_X / 2))]
                    face = cv.resize(face,(roiSize,roiSize))
                    tgt_frame_list.append(face)
                    grayed = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                    grayed = cv.resize(grayed, (roiSize*2,roiSize*2))
                    roi = grayed[int(roiSize-(roiSize/2)):int(roiSize+(roiSize/2)), int(roiSize-(roiSize/2)):int(roiSize+(roiSize/2))]
                    roiSequence.append(roi)
                
                visual = np.asarray(roiSequence)
                visual = visual[0:min_length*self.fps,...]
                if visual.shape[0]<min_length*self.fps:
                    visual = np.pad(visual,((0,min_length*self.fps-visual.shape[0]),(0,0),(0,0)))


                
                # read label info 
                audio_time = line.split(',')[c*6+3].split('_')[-4:]
                audio_time = list(map(int, audio_time))
                label = np.zeros(audio_time[1] - audio_time[0])
                label[audio_time[2] - audio_time[0]: audio_time[3] - audio_time[0]] = 1
                label = label[start:end]
                label = label[:min_length*self.sampling_rate]
                if len(label)<min_length*self.sampling_rate:
                    label = np.pad(label,(0,min_length*self.sampling_rate-len(label)))

                activate_labels.append(label)

                visuals.append(visual)
                audios.append(audio)
                audiofeatures.append(audio_mfcc)


            mixtures.append(mixture)
            mixtures.append(mixture)
            mixfeatures.append(mix_mfcc)
            mixfeatures.append(mix_mfcc)

        np_mixtures = np.asarray(mixtures)
        np_audios = np.asarray(audios)
        np_visuals = np.asarray(visuals)
        np_mix_mfcc = np.asarray(mixfeatures)
        np_active_lables = np.asarray(activate_labels)

        return np_mixtures,np_mix_mfcc,np_audios,np_visuals,np_active_lables

    def __len__(self):
        if self.partition=='train':
            return len(self.minibatch)
        else:
            return len(self.minibatch)



class DistributedSampler(data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=1234):
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
