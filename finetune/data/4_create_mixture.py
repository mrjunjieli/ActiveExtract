import os
import numpy as np 
import argparse
import scipy.io.wavfile as wavfile
import csv
import tqdm
import soundfile as sf 
import librosa
import cv2 as cv 


EPS = np.finfo(float).eps
np.random.seed(0)

def audiowrite(destpath, audio, sample_rate=16000, norm=False, target_level=-25, \
                clipping_threshold=0.99, clip_test=False):
    '''Function to write audio'''

    if clip_test:
        if is_clipped(audio, clipping_threshold=clipping_threshold):
            raise ValueError("Clipping detected in audiowrite()! " + \
                            destpath + " file not written to disk.")

    if norm:
        audio = normalize(audio, target_level)
        max_amp = max(abs(audio))
        if max_amp >= clipping_threshold:
            audio = audio/max_amp * (clipping_threshold-EPS)

    destpath = os.path.abspath(destpath)
    destdir = os.path.dirname(destpath)

    if not os.path.exists(destdir):
        os.makedirs(destdir)

    sf.write(destpath, audio, sample_rate)
    return

def is_clipped(audio, clipping_threshold=0.99):
    return any(abs(audio) > clipping_threshold)

def normalize(audio, target_level=-25):
    '''Normalize the signal to the target level'''
    rms = (audio ** 2).mean() ** 0.5
    scalar = 10 ** (target_level / 20) / (rms+EPS)
    audio = audio * scalar
    return audio

def normalize_segmental_rms(audio, rms, target_level=-25):
    '''Normalize the signal to the target level
    based on segmental RMS'''
    scalar = 10 ** (target_level / 20) / (rms+EPS)
    audio = audio * scalar
    return audio

def audioread(path, norm=False, start=0, stop=None, target_level=-25):
    '''Function to read audio'''
    '''taget_level dBFs'''

    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise ValueError("[{}] does not exist!".format(path))
    try:
        audio, sample_rate = sf.read(path, start=start, stop=stop)
    except RuntimeError:  # fix for sph pcm-embedded shortened v2
        print('WARNING: Audio type not supported')

    if len(audio.shape) == 1:  # mono
        if norm:
            rms = (audio ** 2).mean() ** 0.5
            scalar = 10 ** (target_level / 20) / (rms+EPS)
            audio = audio * scalar
    else:  # multi-channel
        audio = audio.T
        audio = audio.sum(axis=0)/audio.shape[0]
        if norm:
            audio = normalize(audio, target_level)
    return audio, sample_rate



def segmental_snr_mixer(clean, noise, snr,min_option=True, target_level_lower=-35,target_level_upper=-5,target_level=-25, clipping_threshold=0.99):
    '''Function to mix clean speech and noise at various segmental SNR levels'''
    if min_option:
        length = min(len(clean),len(noise))
        if len(clean)>length:
            clean = clean[0:length]
        if len(noise)>length:
            noise = noise[0:length]
    else:
        if len(clean) > len(noise):
            noise = np.append(noise, np.zeros(len(clean)-len(noise)))
        else:
            noise = noise[0:len(clean)]
        
    clean = clean/(max(abs(clean))+EPS)
    noise = noise/(max(abs(noise))+EPS)
    rmsclean, rmsnoise = active_rms(clean=clean, noise=noise)
    
    # Set the noise level for a given SNR
    if rmsclean==0:
        noise = normalize_segmental_rms(noise, rms=rmsnoise, target_level=target_level)
        noisenewlevel=noise
    elif rmsnoise==0:
        noisenewlevel = noise
        clean = normalize_segmental_rms(clean, rms=rmsclean, target_level=target_level)
    else:
        clean = normalize_segmental_rms(clean, rms=rmsclean, target_level=target_level)
        noise = normalize_segmental_rms(noise, rms=rmsnoise, target_level=target_level)
        noisescalar = rmsclean / (10**(snr/20)) / (rmsnoise+EPS)
        noisenewlevel = noise * noisescalar

    # Mix noise and clean speech
    noisyspeech = clean + noisenewlevel
    # Randomly select RMS value between -15 dBFS and -35 dBFS and normalize noisyspeech with that value
    # There is a chance of clipping that might happen with very less probability, which is not a major issue. 
    noisy_rms_level = np.random.randint(target_level_lower, target_level_upper)
    rmsnoisy = (noisyspeech**2).mean()**0.5
    scalarnoisy = 10 ** (noisy_rms_level / 20) / (rmsnoisy+EPS)
    noisyspeech = noisyspeech * scalarnoisy
    clean = clean * scalarnoisy
    noisenewlevel = noisenewlevel * scalarnoisy
    # Final check to see if there are any amplitudes exceeding +/- 1. If so, normalize all the signals accordingly
    if is_clipped(noisyspeech):
        noisyspeech_maxamplevel = max(abs(noisyspeech))/(clipping_threshold-EPS)
        noisyspeech = noisyspeech/noisyspeech_maxamplevel
        clean = clean/noisyspeech_maxamplevel
        noisenewlevel = noisenewlevel/noisyspeech_maxamplevel
        noisy_rms_level = int(20*np.log10(scalarnoisy/noisyspeech_maxamplevel*(rmsnoisy+EPS)))

    return clean, noisenewlevel, noisyspeech, noisy_rms_level


def active_rms(clean, noise, fs=16000, energy_thresh=-50):
    '''Returns the clean and noise RMS of the noise calculated only in the active portions'''
    window_size = 100 # in ms
    window_samples = int(fs*window_size/1000)
    sample_start = 0
    noise_active_segs = []
    clean_active_segs = []

    while sample_start < len(noise):
        sample_end = min(sample_start + window_samples, len(noise))
        noise_win = noise[sample_start:sample_end]
        clean_win = clean[sample_start:sample_end]
        noise_seg_rms = 20*np.log10((noise_win**2).mean()+EPS)
        clean_seg_rms= 20*np.log10((clean_win**2).mean()+EPS)
        # Considering frames with energy
        if noise_seg_rms > energy_thresh:
            noise_active_segs = np.append(noise_active_segs, noise_win)
        if clean_seg_rms>energy_thresh:
            clean_active_segs = np.append(clean_active_segs, clean_win)
        sample_start += window_samples

    if len(noise_active_segs)!=0:
        noise_rms = (noise_active_segs**2).mean()**0.5
    else:
        noise_rms = 0
        
    if len(clean_active_segs)!=0:
        clean_rms = (clean_active_segs**2).mean()**0.5
    else:
        clean_rms = 0

    return clean_rms, noise_rms



def create_mixture(data_list,  data, length, w_file):
	L = 0
	while True:
		mixtures=[data]
		cache = []

		# target speaker
		idx = np.random.randint(0, len(data_list))
		ID =  data_list[idx][0] + data_list[idx][2].split('_')[-5]
		cache.append(ID)

		time = data_list[idx][2].split('_')[-4:]
		time = list(map(int, time))

		tgt_tot_length = time[1] - time[0]

		tgt_length = np.random.randint(min(tgt_tot_length, 16000*3), min(tgt_tot_length +1, 16000*6))
		if tgt_length/16000<3:
			continue
		tgt_start = np.random.randint(0, tgt_tot_length - tgt_length+1)
		tgt_end = tgt_start + tgt_length

		ratio = 0

		mixtures = mixtures + list(data_list[idx]) + [tgt_start,tgt_end,ratio]

		label = np.zeros(tgt_tot_length)
		label[time[2] - time[0]: time[3] - time[0]] = 1
		tgt_label = label[tgt_start:tgt_end]


		# inteference speaker
		while len(cache) < (args.C):
			idx = np.random.randint(0, len(data_list))
			ID =  data_list[idx][0] + data_list[idx][2].split('_')[-5]
			if ID in cache:
				continue
		
			itf_time = data_list[idx][2].split('_')[-4:]
			itf_time = list(map(int, itf_time))
			itf_tot_length = itf_time[1] - itf_time[0]
			if itf_tot_length < tgt_length:
				continue

			cache.append(ID)

			itf_start = np.random.randint(0, itf_tot_length - tgt_length+1)
			itf_end = itf_start + tgt_length


			ratio = float("{:.2f}".format(np.random.uniform(-args.mix_db,args.mix_db)))

			mixtures = mixtures + list(data_list[idx])  + [itf_start,itf_end,ratio]
		
		
		label = np.zeros(itf_tot_length)
		label[itf_time[2] - itf_time[0]: itf_time[3] - itf_time[0]] = 1
		itf_label = label[itf_start:itf_end]


		if np.sum(itf_label*tgt_label)==0: # to avoid too much QQ
			if np.random.uniform(0,1)>0.5:
				continue 
		if np.sum(itf_label)+np.sum(tgt_label)==0:
			continue

		L+=1
		if L>length:
			break
		mixtures.append(round(tgt_length/16000,2))
		w_file.writerow(mixtures)


def main(args):
	##############################
	##############################
	# Get test set list of audios
	test_utts = []
	for path, dirs ,files in os.walk(args.audio_data_direc+'test/'):
		for filename in files:
			if filename[-4:] =='.wav' and filename[0] != '.':
				ln = [path.split('/')[-2],path.split('/')[-1], filename.split('.')[0]]
				test_utts.append(ln)

	# Get val set list of audios
	val_utts = []
	for path, dirs ,files in os.walk(args.audio_data_direc+'val/'):
		for filename in files:
			if filename[-4:] =='.wav' and filename[0] != '.':
				ln = [path.split('/')[-2],path.split('/')[-1], filename.split('.')[0]]
				val_utts.append(ln)

	# Get train set list of audios
	train_utts = []
	for path, dirs ,files in os.walk(args.audio_data_direc+'train/'):
		for filename in files:
			if filename[-4:] =='.wav' and filename[0] != '.':
				ln = [path.split('/')[-2],path.split('/')[-1], filename.split('.')[0]]
				train_utts.append(ln)
	##############################
	##############################


	print("Creating mixture list")
	f_talk=open(args.mixture_data_list,'w')
	w_talk=csv.writer(f_talk)

	create_mixture(test_utts, 'test', args.test_samples, w_talk)
	create_mixture(val_utts, 'val', args.val_samples, w_talk)
	create_mixture(train_utts, 'train', args.train_samples, w_talk)

	return


def create_audio(args):
	# create mixture
	mixture_data_list = open(args.mixture_data_list).read().splitlines()
	print(len(mixture_data_list))

	for line in tqdm.tqdm(mixture_data_list,desc = "Generating audio mixtures"):
		data = line.split(',')
		save_direc=args.mixture_audio_direc+data[0]+'/'
		if not os.path.exists(save_direc+'mix/'):
			os.makedirs(save_direc+'mix/')
		if not os.path.exists(save_direc+'s1/'):
			os.makedirs(save_direc+'s1/')
		if not os.path.exists(save_direc+'s2/'):
			os.makedirs(save_direc+'s2/')
		
		mixture_save_path=save_direc+'mix/'+line.replace(',','_').replace('/','_')+'.wav'

		if os.path.exists(mixture_save_path):
			continue
		
		# read target audio
		c = 0
		audio_clean,_=audioread(args.audio_data_direc+data[0]+'/'+data[c*6+1]+'/'+data[c*6+2]+'/'+data[c*6+3]+'.wav')
		start = int(data[4])
		end = int(data[5])
		audio_clean = audio_clean[start:end]


		# read inteference audio
		c=1
		audio_path=args.audio_data_direc+data[0]+'/'+data[c*6+1]+'/'+data[c*6+2]+'/'+data[c*6+3]+'.wav'
		audio,_ = audioread(audio_path)
		start = int(data[c*6+4])
		end = int(data[c*6+5])
		audio = audio[start:end]
		snr = float(data[c*6+6])
		tgt_audio, int_audio, mix_audio, _ = segmental_snr_mixer(audio_clean,audio,snr)


		s1_path = save_direc+'s1/'+line.replace(',','_').replace('/','_')+'.wav'
		s2_path = save_direc+'s2/'+line.replace(',','_').replace('/','_')+'.wav'
		audiowrite(mixture_save_path, mix_audio)
		audiowrite(s1_path, tgt_audio)
		audiowrite(s2_path, int_audio)



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='iemocap dataset')
	parser.add_argument('--audio_data_direc', default='/workspace2/junjie/dataset/IEMOCAP/uss/clean/' ,type=str)
	parser.add_argument('--mixture_audio_direc', default='/workspace2/junjie/dataset/IEMOCAP/uss/new_mixture/', type=str)
	parser.add_argument('--C', default = 2 ,type=int) 
	parser.add_argument('--mix_db', default = 5, type=float)
	parser.add_argument('--train_samples', default = 20000, type=int)
	parser.add_argument('--val_samples', default = 5000, type=int)
	parser.add_argument('--test_samples', default = 3000, type=int)
	parser.add_argument('--mixture_data_list', default = 'new_mixture_data_list_2mix.csv', type=str)
	args = parser.parse_args()
	
	# create data list
	# main(args)

	# # generate 2 speaker mixture
	create_audio(args)

	# # generate 3 speaker mixture
	# args.mixture_data_list = 'mixture_data_list_3mix.csv'
	# args.C = 3
	# create_audio(args)

	