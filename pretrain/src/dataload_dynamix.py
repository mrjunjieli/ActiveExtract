import os
import numpy as np
# np.set_printoptions(threshold=np.inf)
# import torch 
# torch.set_printoptions(threshold=np.inf)
import python_speech_features
import librosa
import random
import math
import torch.distributed as dist
import torch
import torch.utils.data as data
import os
import cv2
import copy
from tools import audiowrite, audioread, segmental_snr_mixer,normalize


EPS = np.finfo(float).eps
MAX_INT16 = np.iinfo(np.int16).max
np.random.seed(1234)
random.seed(1234)



def generate_wav_list_from_audiodirc(partition,dirpath):
    wav_list = []
    for path, dirs ,files in os.walk(dirpath):
        for filename in files:
            if filename[-4:] =='.wav' and '/'+os.path.join(partition)+'/' in path:
                wav_list.append(os.path.join(path,filename))
                
    return wav_list



class dataset(data.Dataset):
    def __init__(self,
                audio_direc,
                video_direc,
                batch_size,
                partition='train',
                audio_only=False,
                sampling_rate=16000,
                fps = 25,
                max_length=4,
                mix_no=2,
                mix_db =10):

        self.minibatch =[]
        self.audio_only = audio_only
        self.audio_direc = audio_direc
        self.video_direc = video_direc
        self.mix_db = mix_db
        self.sampling_rate = sampling_rate
        self.partition = partition
        self.max_length = max_length
        self.C=mix_no
        self.fps = fps 
        self.batch_size = batch_size


        #read wav list from audio_direc 
        self.audio_list =  generate_wav_list_from_audiodirc(partition,audio_direc)
        
        self.spk_path_dict={}
        for path in self.audio_list:
            spk = path.split('/')[-3]
            if spk not in self.spk_path_dict.keys():
                self.spk_path_dict[spk] = [path]
            else:
                self.spk_path_dict[spk].append(path)
        
        for spk in self.spk_path_dict.keys():
            self.spk_path_dict[spk].sort()

        train_speakers={}
        self.spk_list = list(self.spk_path_dict.keys())
        for  spk in self.spk_list:
            train_speakers[spk] = 0

        #split into train and dev 
        val_list = []
        train_list=[]
        for ln in self.audio_list:
            spk = ln.split('/')[-3]
            if train_speakers[spk] <= 3:
                val_list.append(ln)
                train_speakers[spk] +=1
            else:
                train_list.append(ln)
                train_speakers[spk] +=1

        if partition == 'train':
            wav_list = train_list

        # print(train_list)
        start = 0
        while True:
            end = min(len(wav_list), start + self.batch_size)
            self.minibatch.append(wav_list[start:end])
            if end == len(wav_list):
                break
            start = end
        

    def __getitem__(self, index):
        
        if self.partition=='train' and index==0:
            random.shuffle(self.minibatch)

        batch_lst = self.minibatch[index]
        mixtures=[]
        audios=[]
        visuals=[]
        mixfeatures= []

        for line in batch_lst:
            start = 0
            tgt_audio_path = line
            tgt_audio,sr = audioread(tgt_audio_path,norm=True)
            if sr != self.sampling_rate:
                tgt_audio = librosa.resample(tgt_audio,orig_sr=sr, target_sr=self.sampling_rate)

            tgt_spk = tgt_audio_path.split('/')[-3]

            if self.partition=='train':
                
                #select inteference speaker 
                left_spk_list = copy.deepcopy(self.spk_list)
                left_spk_list.remove(tgt_spk)
                intervention_audio_spk = np.random.choice(left_spk_list)
                intervention_audio_path = np.random.choice(self.spk_path_dict[intervention_audio_spk])
                intervention_audio,sr = audioread(intervention_audio_path,norm=True)
                if sr != self.sampling_rate:
                    intervention_audio = librosa.resample(intervention_audio,orig_sr=sr, target_sr=self.sampling_rate)
                intervention_audio = normalize(intervention_audio)
                snr = round(np.random.uniform(-self.mix_db,self.mix_db),2)

            tgt_audio, intervention_audio, mix_audio, _ = segmental_snr_mixer(tgt_audio,intervention_audio,snr)


            if len(tgt_audio)//self.sampling_rate >=self.max_length:
                if self.partition=='train':
                    start = random.randint(0,len(tgt_audio)//self.sampling_rate-self.max_length)
                    if start-1>=0:
                        start = start-1
                tgt_audio = tgt_audio[start*self.sampling_rate:(start+self.max_length)* self.sampling_rate]
                intervention_audio = intervention_audio[start*self.sampling_rate:(start+self.max_length)* self.sampling_rate]
                mix_audio = mix_audio[start*self.sampling_rate:(start+self.max_length)* self.sampling_rate]
            else:
                tgt_audio = np.pad(tgt_audio, ((0, int(sr* self.max_length - len(tgt_audio)))), mode = 'edge')
                intervention_audio = np.pad(intervention_audio, ((0, int(sr* self.max_length - len(intervention_audio)))), mode = 'edge')
                mix_audio = np.pad(mix_audio, ((0, int(sr* self.max_length - len(mix_audio)))), mode = 'edge')

            #generate mfcc of mix_audio 
            # fps is not always 25, in order to align the visual, we modify the window and step in MFCC extraction process based on fps
            mix_mfcc = python_speech_features.mfcc((mix_audio*MAX_INT16).astype(np.int16), self.sampling_rate, numcep = 13, winlen = 0.025 * 25 / self.fps, winstep = 0.010 * 25 / self.fps)
            if mix_mfcc.shape[0]<self.max_length*self.fps*4:
                mix_mfcc = np.pad(mix_mfcc, ((0, (self.max_length*self.fps*4) -mix_mfcc.shape[0]), (0,0)), 'edge')
            


            #load tgt video 
            tgt_audio_path_part = '/'.join(tgt_audio_path.split('/')[-4:])
            tgt_audio_path_part = tgt_audio_path_part.split('.')[0]+'.mp4'
            video_path = os.path.join(self.video_direc,tgt_audio_path_part)

            captureObj = cv2.VideoCapture(video_path)
            roiSequence = []
            roiSize = 112

            while (captureObj.isOpened()):
                ret, frame = captureObj.read()
                if ret == True:

                    grayed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    grayed = cv2.resize(grayed, (roiSize * 2, roiSize * 2))
                    roi = grayed[int(roiSize - (roiSize / 2)):int(roiSize + (roiSize / 2)),
                        int(roiSize - (roiSize / 2)):int(roiSize + (roiSize / 2))]
                    roiSequence.append(roi)
                else:
                    break
            captureObj.release()


            tgt_visual = np.asarray(roiSequence) 


            #read interference video
            intervention_audio_path_part = '/'.join(intervention_audio_path.split('/')[-4:])
            intervention_audio_path_part = intervention_audio_path_part.split('.')[0]+'.mp4'
            video_path = os.path.join(self.video_direc,intervention_audio_path_part)

            captureObj = cv2.VideoCapture(video_path)
            roiSequence = []
            roiSize = 112

            while (captureObj.isOpened()):
                ret, frame = captureObj.read()
                if ret == True:
                    grayed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    grayed = cv2.resize(grayed, (roiSize * 2, roiSize * 2))
                    roi = grayed[int(roiSize - (roiSize / 2)):int(roiSize + (roiSize / 2)),
                        int(roiSize - (roiSize / 2)):int(roiSize + (roiSize / 2))]

                    roiSequence.append(roi)
                else:
                    break
            captureObj.release()
            int_visual = np.asarray(roiSequence) 


            if tgt_visual.shape[0]//self.fps>=self.max_length:
                tgt_visual= tgt_visual[start*self.fps:(start+self.max_length)*self.fps,...]
            else:
                tgt_visual = np.pad(tgt_visual, ((0, int(self.max_length*self.fps - tgt_visual.shape[0])), (0,0), (0,0)), mode = 'edge')

            if int_visual.shape[0]//self.fps>=self.max_length:
                int_visual= int_visual[start*self.fps:(start+self.max_length)*self.fps,...]
            else:
                int_visual = np.pad(int_visual, ((0, int(self.max_length*self.fps - int_visual.shape[0])), (0,0), (0,0)), mode = 'edge')

            mixtures.append(mix_audio)
            mixtures.append(mix_audio)

            audios.append(tgt_audio)
            audios.append(intervention_audio)


            visuals.append(tgt_visual)
            visuals.append(int_visual)

            mixfeatures.append(mix_mfcc)
            mixfeatures.append(mix_mfcc)
        
        np_mixtures = np.asarray(mixtures)
        np_audios = np.asarray(audios)
        np_visuals = np.asarray(visuals)
        np_mix_mfcc = np.asarray(mixfeatures)

        return np_mixtures,np_mix_mfcc,np_audios,np_visuals

    def __len__(self):
        
        if self.partition=='train':
            return len(self.minibatch)//40
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



def get_dataloader_dy(args, partition):
    datasets = dataset(
                audio_direc=args.audio_direc,
                video_direc=args.visual_direc,
                batch_size=args.batch_size,
                max_length=args.max_length,
                partition=partition,
                mix_no=args.C)
    sampler = DistributedSampler(
        datasets,
        num_replicas=args.world_size,
        rank=args.local_rank) if args.distributed else None

    generator = data.DataLoader(datasets,
            batch_size = 1,
            shuffle = (sampler is None),
            num_workers = args.num_workers,
            sampler=sampler)

    return sampler, generator


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser("av-yolo")
    
    # Dataloader
    parser.add_argument('--audio_direc', type=str, default='/workspace2/junjie/dataset/voxceleb2/wav/')
    parser.add_argument('--visual_direc', type=str, default='/workspace2/junjie/dataset/voxceleb2/mp4/')
    parser.add_argument('--max_length', default=4, type=int)
    
    # Training    
    parser.add_argument('--batch_size', default=12, type=int,
                        help='Batch size')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers to generate minibatch')
    parser.add_argument('--epochs', default=30, type=int,
                        help='Number of maximum epochs')

    # Model hyperparameters
    parser.add_argument('--L', default=40, type=int,
                        help='Length of the filters in samples (80=5ms at 16kHZ)')
    parser.add_argument('--N', default=256, type=int,
                        help='Number of filters in autoencoder')
    parser.add_argument('--C', type=int, default=2,
                        help='number of speakers to mix')

    # optimizer
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Init learning rate')
    parser.add_argument('--max_norm', default=5, type=float,
                    help='Gradient norm threshold to clip')


    # Log and Visulization
    parser.add_argument('--log_name', type=str, default='test',
                        help='the name of the log')
    parser.add_argument('--use_tensorboard', type=int, default=0,
                        help='Whether to use use_tensorboard')
    parser.add_argument('--continue_from', type=str, default='',
                        help='Whether to resume training')

    # Distributed training
    parser.add_argument("--local_rank", default=0, type=int)

    args = parser.parse_args()

    args.distributed = True
    args.world_size = 1
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        args.world_size = int(os.environ['WORLD_SIZE'])


    train_sampler, train_generator = get_dataloader_dy(args, 'train')
    
    mix_total = 0
    save_folder = 'tempory_folder'


    for i, (a_mix,a_mix_mfcc,audio,visual) in enumerate(train_generator):
        print(a_mix.shape) # 1,4,96000
        print(a_mix_mfcc.shape) # 1,4,600,13 
        print(audio.shape) # 
        print(visual.shape) # 1,4,150,112,112
        # print(face.shape) #1,4,112,112,3
        # pass
        for i in range(4):
            # a_tgt = audio.squeeze()[i,:].cpu().numpy()
            # mix = a_mix.squeeze()[i,:].cpu().numpy()
            # audiowrite('./%d_tgt.wav'%i,a_tgt)
            # audiowrite('./%d_output.wav'%i,mix)
            for d in range(visual.shape[2]):
                cv2.imwrite('./%d_%dtest.jpg'%(i,d),visual[0][i][d].cpu().numpy())
                if d>=3:
                    break
        
        # os.makedirs(save_folder,exist_ok=True)
        # for i in range(face.shape[1]):
        #     face_ = face[0][i].cpu().numpy()
        #     roiSize_Y = 135
        #     roiSize_X = 125
        #     face_ = face_[0:roiSize_Y,int(112 - (roiSize_X / 2)):int(112 + (roiSize_X / 2))]
        #     cv2.imwrite(save_folder+'/%d_test_crop.jpg'%(i),face_)
        #     cv2.imwrite(save_folder+'/%d_test.jpg'%(i),face[0][i].cpu().numpy())

        # break
