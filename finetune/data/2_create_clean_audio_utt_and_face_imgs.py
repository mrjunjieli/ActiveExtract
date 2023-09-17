
import os
import numpy as np 
import argparse
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



def write_npy(fname, file):
	fdir = os.path.dirname(fname)
	if fdir and not os.path.exists(fdir):
		os.makedirs(fdir)
	np.save(fname,file)


def visual_images(visual_path):
	captureObj = cv.VideoCapture(visual_path)
	roiSequence = list()
	while (captureObj.isOpened()):
		ret, frame = captureObj.read()
		if ret == True:
			roiSequence.append(frame)
		else:
			break
	captureObj.release()
	img = np.stack(roiSequence, axis=0)
	return img


def main(args):
	# Get test set list of audios
	talks = []
	for path, dirs ,files in os.walk(args.audio_data_direc):
		for filename in files:
			if filename[-4:] =='.wav' and filename[0] != '.':
				ln = [path.split('/')[-2],path.split('/')[-1], filename.split('.')[0]]
				talks.append(ln)
				# print(ln)

	np.random.shuffle(talks)
	train_talks = talks[:240]
	val_talks = talks[240:270]
	test_talks = talks[270:]

	train_utts = []
	for line in train_talks:
		text_path = args.text_data_direc + line[0]+'/'+line[1]+'.txt'
		text_lst =open(text_path).read().splitlines()
		text_lst=list(filter(lambda x: x.startswith(line[2]), text_lst))

		audio,_=audioread(args.audio_data_direc+line[0]+'/'+line[1]+'/'+line[2]+'.wav')
		visual_path = args.video_data_direc+line[0]+'/'+line[1]+'/'+line[2]+'.avi'
		images = visual_images(visual_path)

		# cut from the random point of the silence region
		for i, (seg) in enumerate(text_lst):

			time = seg.split(' ')[1]
			start_time = round(float(time.split('-')[0][1:])*16000)-320
			end_time = round(float(time.split('-')[1][:-2])*16000)-320

			if end_time <= start_time: continue
			if audio.shape[0] < start_time: continue
			if audio.shape[0] < end_time: continue

			if i == 0:
				pre_end_time = start_time
			else:
				pre_end_time = nex_start_time

			assert pre_end_time <= start_time

			if i == (len(text_lst) -1):
				nex_start_time = end_time
			else:
				nex_time = text_lst[i+1].split(' ')[1]
				nex_start_time = round(float(nex_time.split('-')[0][1:])*16000)-320 
				if nex_start_time < end_time: continue
				if nex_start_time != end_time: nex_start_time = np.random.randint(end_time, nex_start_time)
			
			ln_append = line + [str(pre_end_time)] + [str(nex_start_time)] + [str(start_time)] + [str(end_time)]
			train_utts.append(ln_append)

			audio_save_path = args.clean_audio_data_direc + 'train/' + '/'.join(line[:-1]) + '/' + '_'.join(ln_append[2:]) + '.wav'
			audio_tgt = audio[pre_end_time:nex_start_time]
			# audiowrite(audio_save_path , audio_tgt)

			v_pre_end_time = round(pre_end_time/16000*25)
			v_nex_start_time = round(nex_start_time/16000*25)
			visual_img = images[v_pre_end_time:v_nex_start_time]
			visual_save_path = args.visual_frame_direc + 'train/' + '/'.join(line[:-1]) + '/' + '_'.join(ln_append[2:]) + '.npy'
			# write_npy(visual_save_path,visual_img)


	val_utts = []
	for line in val_talks:
		text_path = args.text_data_direc + line[0]+'/'+line[1]+'.txt'
		text_lst =open(text_path).read().splitlines()
		text_lst=list(filter(lambda x: x.startswith(line[2]), text_lst))

		audio,_=audioread(args.audio_data_direc+line[0]+'/'+line[1]+'/'+line[2]+'.wav')
		visual_path = args.video_data_direc+line[0]+'/'+line[1]+'/'+line[2]+'.avi'
		# images = visual_images(visual_path)

		# cut from the random point of the silence region
		for i, (seg) in enumerate(text_lst):

			time = seg.split(' ')[1]
			start_time = round(float(time.split('-')[0][1:])*16000)-320
			end_time = round(float(time.split('-')[1][:-2])*16000)-320

			if end_time <= start_time: continue
			if audio.shape[0] < start_time: continue
			if audio.shape[0] < end_time: continue

			if i == 0:
				pre_end_time = start_time
			else:
				pre_end_time = nex_start_time

			assert pre_end_time <= start_time

			if i == (len(text_lst) -1):
				nex_start_time = end_time
			else:
				nex_time = text_lst[i+1].split(' ')[1]
				nex_start_time = round(float(nex_time.split('-')[0][1:])*16000)-320 
				if nex_start_time < end_time: continue
				if nex_start_time != end_time: nex_start_time = np.random.randint(end_time, nex_start_time)
			
			ln_append = line + [str(pre_end_time)] + [str(nex_start_time)] + [str(start_time)] + [str(end_time)]
			val_utts.append(ln_append)

			audio_save_path = args.clean_audio_data_direc + 'val/' + '/'.join(line[:-1]) + '/' + '_'.join(ln_append[2:]) + '.wav'
			audio_tgt = audio[pre_end_time:nex_start_time]
			# audiowrite(audio_save_path , audio_tgt)

			v_pre_end_time = round(pre_end_time/16000*25)
			v_nex_start_time = round(nex_start_time/16000*25)
			visual_img = images[v_pre_end_time:v_nex_start_time]
			visual_save_path = args.visual_frame_direc + 'val/' + '/'.join(line[:-1]) + '/' + '_'.join(ln_append[2:]) + '.npy'
			# write_npy(visual_save_path,visual_img)


	test_utts = []
	for line in test_talks:
		text_path = args.text_data_direc + line[0]+'/'+line[1]+'.txt'
		text_lst =open(text_path).read().splitlines()
		text_lst=list(filter(lambda x: x.startswith(line[2]), text_lst))

		audio,_=audioread(args.audio_data_direc+line[0]+'/'+line[1]+'/'+line[2]+'.wav')
		visual_path = args.video_data_direc+line[0]+'/'+line[1]+'/'+line[2]+'.avi'
		# images = visual_images(visual_path)


		# cut from the random point of the silence region
		for i, (seg) in enumerate(text_lst):

			time = seg.split(' ')[1]
			start_time = round(float(time.split('-')[0][1:])*16000)-320
			end_time = round(float(time.split('-')[1][:-2])*16000)-320

			if end_time <= start_time: continue
			if audio.shape[0] < start_time: continue
			if audio.shape[0] < end_time: continue

			if i == 0:
				pre_end_time = start_time
			else:
				pre_end_time = nex_start_time

			assert pre_end_time <= start_time

			if i == (len(text_lst) -1):
				nex_start_time = end_time
			else:
				nex_time = text_lst[i+1].split(' ')[1]
				nex_start_time = round(float(nex_time.split('-')[0][1:])*16000)-320 
				if nex_start_time < end_time: continue
				if nex_start_time != end_time: nex_start_time = np.random.randint(end_time, nex_start_time)
			
			ln_append = line + [str(pre_end_time)] + [str(nex_start_time)] + [str(start_time)] + [str(end_time)]
			test_utts.append(ln_append)

			audio_save_path = args.clean_audio_data_direc + 'test/' + '/'.join(line[:-1]) + '/' + '_'.join(ln_append[2:]) + '.wav'
			audio_tgt = audio[pre_end_time:nex_start_time]
			# audiowrite(audio_save_path , audio_tgt)

			v_pre_end_time = round(pre_end_time/16000*25)
			v_nex_start_time = round(nex_start_time/16000*25)
			visual_img = images[v_pre_end_time:v_nex_start_time]
			visual_save_path = args.visual_frame_direc + 'test/' + '/'.join(line[:-1]) + '/' + '_'.join(ln_append[2:]) + '.npy'
			# write_npy(visual_save_path,visual_img)

	print(len(test_utts))
	print(len(val_utts))
	print(len(train_utts))

	return

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='iemocap dataset')
	parser.add_argument('--audio_data_direc', default='/workspace2/junjie/dataset/IEMOCAP/audio_clean/' ,type=str)
	parser.add_argument('--text_data_direc', default='/workspace2/junjie/dataset/IEMOCAP/transcriptions/' ,type=str)
	parser.add_argument('--clean_audio_data_direc', default='/workspace2/junjie/dataset/IEMOCAP/uss/clean/' ,type=str)
	parser.add_argument('--video_data_direc', default='/workspace2/junjie/dataset/IEMOCAP/face_crop/', type=str)
	parser.add_argument('--visual_frame_direc',default ='/workspace2/junjie/dataset/IEMOCAP/uss/face_npy/', type=str)
	args = parser.parse_args()
	
	main(args)