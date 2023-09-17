
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys 
sys.path.append('/workspace2/junjie/ASD_SE/model/')
# from spkeaker_emb.campplus import CAMPPlus
# from spkeaker_emb.tools import parse_config_or_kwargs
# import torchaudio.compliance.kaldi as Kaldi
import numpy as np 
import random 


EPS = 1e-6


class BaseLoss(nn.Module):
    def __init__(self):
        super(BaseLoss, self).__init__()

    def forward(self, preds, targets, weight=None,**kwargs):
        if isinstance(preds, list):
            N = len(preds)
            if weight is None:
                weight = preds[0].new_ones(1)

            errs = [self._forward(preds[n], targets[n], weight[n],**kwargs)
                    for n in range(N)]
            err = torch.mean(torch.stack(errs))

        elif isinstance(preds, torch.Tensor):
            if weight is None:
                weight = preds.new_ones(1)
            err = self._forward(preds, targets, weight,**kwargs)
        return err

# class TripletLossCosine(BaseLoss):
#     """
#     Triplet loss with cosine distance
#     Takes embeddings of an anchor sample, a positive sample and a negative sample
#     """

#     def __init__(self, margin=0.5,\
#         speaker_pretrained_model = './spkeaker_emb/avg_model.pt',\
#         config = './spkeaker_emb/config.yaml'):
#         super(TripletLossCosine, self).__init__()
#         self.margin = margin
#         #speaker_encoder
#         configs = parse_config_or_kwargs(config,model_path=speaker_pretrained_model)
#         model_path = configs['model_path']
#         self.speaker_encoder = CAMPPlus(**configs['model_args']) 
#         self.speaker_encoder.cuda()  
#         pretrained_model = torch.load(model_path, map_location='cpu')
#         state =self.speaker_encoder.state_dict()
#         for key in state.keys():
#             pretrain_key = key
#             if pretrain_key in pretrained_model.keys():
#                 state[key] = pretrained_model[pretrain_key]
#             else:
#                 print("not %s loaded" % pretrain_key)
#         self.speaker_encoder.load_state_dict(state)
#         self.speaker_encoder.eval()

#     def _forward(self, estimate,target,weight=None, sample_rate=16000,size_average=True,gt_pro=0.2):
#         assert estimate.shape[0]%4==0
#         losses =0
#         for i in range(estimate.shape[0]//4):
#             spk_A1 = estimate[i,:].unsqueeze(0)
#             spk_B = estimate[i+1,:].unsqueeze(0)
#             spk_A2 = estimate[i+2,:].unsqueeze(0)
#             spk_C = estimate[i+3,:].unsqueeze(0)


#             gt_spk_B = target[i+1,:].unsqueeze(0)
#             gt_spk_A2 = target[i+2,:].unsqueeze(0)
#             gt_spk_C = target[i+3,:].unsqueeze(0)

#             spk_A1_fbank = Kaldi.fbank(spk_A1, num_mel_bins=80,sample_frequency=sample_rate)
#             spk_B_fbank = Kaldi.fbank(spk_B, num_mel_bins=80,sample_frequency=sample_rate)
#             spk_A2_fbank = Kaldi.fbank(spk_A2, num_mel_bins=80,sample_frequency=sample_rate)
#             spk_C_fbank = Kaldi.fbank(spk_C, num_mel_bins=80,sample_frequency=sample_rate)

#             gt_spk_B_fbank = Kaldi.fbank(gt_spk_B, num_mel_bins=80,sample_frequency=sample_rate)
#             gt_spk_A2_fbank = Kaldi.fbank(gt_spk_A2, num_mel_bins=80,sample_frequency=sample_rate)
#             gt_spk_C_fbank = Kaldi.fbank(gt_spk_C, num_mel_bins=80,sample_frequency=sample_rate)

#             with torch.no_grad():
#                 spk_A1_emb = self.speaker_encoder(spk_A1_fbank.unsqueeze(0))
#                 spk_B_emb = self.speaker_encoder(spk_B_fbank.unsqueeze(0))
#                 spk_A2_emb = self.speaker_encoder(spk_A2_fbank.unsqueeze(0))
#                 spk_C_emb = self.speaker_encoder(spk_C_fbank.unsqueeze(0))

#                 gt_spk_B_emb = self.speaker_encoder(gt_spk_B_fbank.unsqueeze(0))
#                 gt_spk_A2_emb = self.speaker_encoder(gt_spk_A2_fbank.unsqueeze(0))
#                 gt_spk_C_emb = self.speaker_encoder(gt_spk_C_fbank.unsqueeze(0))

#             if random.uniform(0,1)<gt_pro:
#                 spk_A2_emb = gt_spk_A2_emb
#                 spk_B_emb = gt_spk_B_emb
#                 spk_C_emb = gt_spk_C_emb

#             distance_positive = 1 - F.cosine_similarity(spk_A1_emb, spk_A2_emb)
#             distance_negative= 1 - 0.5*(F.cosine_similarity(spk_A1_emb, spk_B_emb)+F.cosine_similarity(spk_A1_emb, spk_C_emb))

#             losses=losses +F.relu((distance_positive - distance_negative) + self.margin)[0]
            
        
#         return losses/(estimate.shape[0]//4) if size_average else losses


class SA_SDR(BaseLoss):
    '''
    implement of paper: SA-SDR: A NOVEL LOSS FUNCTION FOR SEPARATION OF MEETING STYLE DATA
    '''
    def __init__(self):
        super(SA_SDR,self).__init__()
    def _forward(self,estimate,target,weight=None):
        target_energy = torch.norm(target, dim=-1) ** 2
        error_energy = torch.norm(estimate - target, dim=-1) ** 2
        target_energy = torch.sum(target_energy, dim=-1)
        error_energy = torch.sum(error_energy, dim=-1)
        sa_sdr = target_energy / (error_energy)
        sa_sdr = 10 * torch.log10(sa_sdr)

        return sa_sdr

class SDR(BaseLoss):
    def __init__(self):
        super(SDR,self).__init__()
    def _forward(self,estimate,target,weight=None):
        assert target.size() == estimate.size()
    
        noise = target - estimate
        ratio = torch.sum(target ** 2, axis = -1) / (torch.sum(noise ** 2, axis = -1) + EPS)
        sdr = 10 * torch.log10(ratio + EPS)

        return torch.mean(sdr)



class SI_SNR(BaseLoss):
    def __init__(self):
        super(SI_SNR,self).__init__()
    
    def _forward(self,estimate,target,weight=None):

        assert target.size() == estimate.size()

        # Step 1. Zero-mean norm
        #I reckon that this step is wrong, cause when estimate is absolutely equal with target, 
        #the result of this function is very low 
        # so i annotate these tow lines
        # target = target - torch.mean(target, axis = -1, keepdim=True)
        # estimate = estimate - torch.mean(estimate, axis = -1, keepdim=True)

        # Step 2. SI-SNR
        # s_target = <s', s>s / ||s||^2
        ref_energy = torch.sum(target ** 2, axis = -1, keepdim=True) + EPS
        proj = torch.sum(target * estimate, axis = -1, keepdim=True) * target / ref_energy
        # e_noise = s' - s_target
        noise = estimate - proj
        # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
        ratio = torch.sum(proj ** 2, axis = -1) / (torch.sum(noise ** 2, axis = -1) + EPS)
        sisnr = 10 * torch.log10(ratio + EPS)

        return torch.mean(sisnr)







if __name__=='__main__':
    q = TripletLossCosine()
    y = SA_SDR()
    x = SI_SNR()
    z = SDR()
    # est = torch.randn(8,64000)
    # pre = torch.randn(8,64000)
    # print(F.cosine_similarity(est, pre))
    # print(x(est,pre))
    # print(y(est,pre))


    import soundfile as sf 
    audio1, sample_rate = sf.read('/workspace2/junjie/reentry_5k/src/reentry_training/tempory_folder/0_0_output.wav')
    audio2, sample_rate = sf.read('/workspace2/junjie/reentry_5k/src/reentry_training/tempory_folder/1_0_tgt.wav')
    audio1 = torch.from_numpy(audio1).unsqueeze(0)
    audio2 = torch.from_numpy(audio2).unsqueeze(0)
    print(x(audio1,audio1))
    print(y(audio1,audio1))
    print(z(audio1,audio1))
    print(q(audio1,audio1))
