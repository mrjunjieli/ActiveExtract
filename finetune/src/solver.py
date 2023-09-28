from pickletools import optimize
import time
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.nn as nn 
from datetime import datetime
import torch 
from model.loss import SA_SDR,SI_SNR
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
import copy
import numpy as np 

EPS = np.finfo(float).eps


scaler = GradScaler()


class Solver(object):
    def __init__(self, train_data, validation_data, model, optimizer, args):
        self.train_data = train_data
        self.validation_data = validation_data
        self.args = args

        self.sisnr = SI_SNR()
        self.sasdr  = SA_SDR()

        self.print = False
        if (self.args.distributed and self.args.local_rank ==0) or not self.args.distributed:
            self.print = True
            if self.args.use_tensorboard:
                self.writer = SummaryWriter('logs/%s/tensorboard/' % args.log_name)

        self.model = model 
        self.optimizer = optimizer


        if self.args.distributed:
            self.model = DDP(self.model, find_unused_parameters=True)

        self._reset()

    def _reset(self):
        self.halving = False
        if self.args.continue_from:
            checkpoint = torch.load('logs/%s/model_dict_last.pt' % self.args.continue_from, map_location='cpu')
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            self.start_epoch=checkpoint['epoch']
            self.best_val_loss = checkpoint['best_val_loss']
            self.val_no_impv = checkpoint['val_no_impv']

            if self.print: print("Resume training from epoch: {}".format(self.start_epoch))
            
        else:
            self.best_val_loss = float("inf")
            self.val_no_impv = 0
            self.start_epoch=1
            if self.print: print('Start new training')

    def train(self):
        for epoch in range(self.start_epoch, self.args.epochs+1):
            self.joint_loss_weight=epoch
            if self.args.distributed: self.args.train_sampler.set_epoch(epoch)

#             Train
            self.model.train()
            start = time.time()
            tr_loss = self._run_one_epoch(data_loader = self.train_data, state='train',epoch=epoch)
            reduced_tr_loss = self._reduce_tensor(tr_loss)

            if self.print: print('Train Summary | End of Epoch {0} | Time {1:.2f}s | Current time {2} |'
                      'Train Loss {3:.3f}| '.format(
                        epoch, time.time() - start,datetime.now(),reduced_tr_loss))

            #Validation
            self.model.eval()
            start = time.time()
            with torch.no_grad():
                val_loss = self._run_one_epoch(data_loader = self.validation_data, state='val')
                reduced_val_loss = self._reduce_tensor(val_loss)
                if self.print: print('Valid Summary | End of Epoch {0} | Time {1:.2f}s | Current time {2} |'
                        'Valid Loss {3:.3f}| '.format(
                            epoch, time.time() - start, datetime.now(),reduced_val_loss))
            
            # Check whether to adjust learning rate and early stop
            find_best_model = False
            if reduced_val_loss >= self.best_val_loss:
                self.val_no_impv += 1
                if self.val_no_impv >= 10:
                    if self.print: print("No imporvement for 10 epochs, early stopping.")
                    break
            else:
                self.val_no_impv = 0
                self.best_val_loss = reduced_val_loss
                find_best_model=True

            if self.val_no_impv == 3:
                self.halving = True

            # Halfing the learning rate
            if self.halving:
                optim_state = self.optimizer.state_dict()
                optim_state['param_groups'][0]['lr'] = optim_state['param_groups'][0]['lr'] /2
                self.optimizer.load_state_dict(optim_state)
                if self.print: print('Learning rate adjusted to: {lr:.6f}'.format(
                    lr=optim_state['param_groups'][0]['lr']))
                self.halving = False

            if self.print:
                # Tensorboard logging
                if self.args.use_tensorboard:
                    self.writer.add_scalar('Train_loss', reduced_tr_loss, epoch)
                    self.writer.add_scalar('Validation_loss', reduced_val_loss, epoch)

                # Save model
                checkpoint = {'model': self.model.state_dict(),
                                'optimizer': self.optimizer.state_dict(),
                                # 'amp': self.amp.state_dict(),
                                'epoch': epoch+1,
                                'best_val_loss': self.best_val_loss,
                                'val_no_impv': self.val_no_impv}
                torch.save(checkpoint, "logs/"+ self.args.log_name+"/model_dict_last.pt")
                if find_best_model:
                    torch.save(checkpoint, "logs/"+ self.args.log_name+"/model_dict_best.pt")
                    print("Fund new best model, dict saved")
                if epoch %10 ==0:
                    torch.save(checkpoint, "logs/"+ self.args.log_name+"/model_dict_"+str(epoch)+".pt")


    def _run_one_epoch(self, data_loader, state,epoch=0):
        step=0
        total_step = len(data_loader)
        total_loss = 0

        self.accu_count = 0
        avg_sisnri_2 = [] 
        avg_sisnri_3 = [] 
        avg_energy_1 = [] 
        avg_energy_4 = []
        
        self.optimizer.zero_grad()
        for i, (a_mix,a_mix_mfcc,a_tgt,v_tgt,label) in enumerate(data_loader):
            a_mix = a_mix.cuda().squeeze(0).float()
            a_tgt = a_tgt.cuda().squeeze(0).float()
            v_tgt = v_tgt.cuda().squeeze(0).float()
            a_mix_mfcc = a_mix_mfcc.cuda().squeeze(0).float()
            itf_label = copy.deepcopy(label)
            for k in range(0,label.shape[1],2):
                itf_label[0,0+k] = label[0,1+k]
                itf_label[0,1+k] = label[0,0+k]

            label = label.cuda().squeeze(0).long()
            itf_label = itf_label.cuda().squeeze(0).long()

            with autocast():
                est_a_tgt = self.model(a_mix,a_mix_mfcc,v_tgt)

                loss = 0
                for j in range(label.shape[0]):
                    a_mix_utt, a_tgt_utt, a_est_utt = self.segment_utt(a_mix[j], a_tgt[j], est_a_tgt[j], label[j], itf_label[j])


                    energy_1, sisnr_2, sisnr_3, energy_4  = self.eval_segment_utt(a_mix_utt, a_tgt_utt,a_est_utt,state)

                    if energy_1:
                        avg_energy_1.append(energy_1.data.cpu().detach().numpy()) #QQ
                        loss += energy_1*0.0005
                    if sisnr_2:
                        avg_sisnri_2.append(sisnr_2.data.cpu().detach().numpy()) #SQ
                        loss += 0.1*sisnr_2
                    if sisnr_3:
                        avg_sisnri_3.append(sisnr_3.data.cpu().detach().numpy()) #SS
                        loss += sisnr_3
                    if energy_4:
                        avg_energy_4.append(energy_4.data.cpu().detach().numpy()) #QS
                        loss += energy_4*0.005
                loss = loss / label.shape[0]

            if state =='train':
                
                self.accu_count += 1
                
                step+=1
                total_loss+=loss.data

                if self.args.accu_grad:
                    loss = loss/(self.args.effec_batch_size / self.args.batch_size)
                    scaler.scale(loss).backward()
                    if self.accu_count == (self.args.effec_batch_size / self.args.batch_size):
                        scaler.step(self.optimizer)
                        scaler.update()
                        self.optimizer.zero_grad()
                        self.accu_count = 0
                else:
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad()
                if i%1000==0:
                    print('step:{}/{} avg loss:{:.3f}, QQ_power:{:.3f},SQ_SDR:{:.3f},SS_SDR:{:.3f},QS_power:{:.3f}'.format(step, total_step,\
                    total_loss / (i+1), np.mean(avg_energy_1) if len(avg_energy_1)>0 else 0, -np.mean(avg_sisnri_2) if len(avg_sisnri_2)>0 else 0, \
                         -np.mean(avg_sisnri_3) if len(avg_sisnri_3)>0 else 0,np.mean(avg_energy_4) if len(avg_energy_4)>0 else 0))

            else: 
                step+=1
                total_loss+=loss.data

        return total_loss / (i+1)

    def _reduce_tensor(self, tensor):
        if not self.args.distributed: return tensor
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= self.args.world_size
        return rt


    def segment_utt(self,a_mix, a_tgt, a_est, label_tgt, label_int):
        utt_list = []
        for utt in [a_mix, a_tgt, a_est]:
            utt_1 = utt[((label_tgt == label_int) & (label_tgt == 0))] #QQ
            utt_3 = utt[((label_tgt == label_int) & (label_tgt == 1))] #SS
            utt_2 = utt[((label_tgt != label_int) & (label_tgt == 1))] #SQ
            utt_4 = utt[((label_tgt != label_int) & (label_tgt == 0))] #QS
            assert utt.shape[-1] == (utt_1.shape[-1]+ utt_2.shape[-1]+ utt_3.shape[-1]+ utt_4.shape[-1])
            utt_list.append([utt_1,utt_2,utt_3,utt_4])  #qq #sq #ss #qs
        return utt_list[0], utt_list[1], utt_list[2]


    def cal_SDR(self,source, estimate_source):
        assert source.size() == estimate_source.size()
        
        estimate_source += EPS # the estimated source is zero sometimes

        noise = source - estimate_source
        ratio = torch.sum(source ** 2, axis = -1) / (torch.sum(noise ** 2, axis = -1) + EPS)
        sdr = 10 * torch.log10(ratio + EPS)

        return sdr

    def cal_logEnergy(self,source):
        ratio = torch.sum(source ** 2, axis = -1)
        sdr = 10 * torch.log10(ratio + EPS)
        return sdr

    def eval_segment_utt(self,a_mix, a_tgt,a_est,state):
        a_mix_1, a_mix_2, a_mix_3, a_mix_4= a_mix[0], a_mix[1], a_mix[2], a_mix[3]
        a_tgt_1, a_tgt_2, a_tgt_3, a_tgt_4= a_tgt[0], a_tgt[1], a_tgt[2], a_tgt[3]
        a_est_1, a_est_2, a_est_3, a_est_4= a_est[0], a_est[1], a_est[2], a_est[3]

        energy_1, sisnr_2, sisnr_3, energy_4 = None, None, None, None

        if a_mix_1.shape[-1]!=0:
            energy_1 = self.cal_logEnergy(a_est_1)

        if a_mix_2.shape[-1]!=0:
            sisnr_2 = - self.cal_SDR(a_tgt_2, a_est_2)

        if a_mix_3.shape[-1]!=0:
            sisnr_3 = - self.cal_SDR(a_tgt_3, a_est_3)

        if a_mix_4.shape[-1]!=0:
            energy_4 = self.cal_logEnergy(a_est_4)

        return energy_1, sisnr_2, sisnr_3, energy_4 