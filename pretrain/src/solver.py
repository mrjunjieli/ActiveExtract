from pickletools import optimize
import time
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.nn as nn 
from datetime import datetime
import torch 
from model.loss import SA_SDR,SI_SNR,SDR
import math

from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP



scaler = GradScaler()


class Solver(object):
    def __init__(self, train_data, validation_data, model, optimizer, args):
        self.train_data = train_data
        self.validation_data = validation_data
        self.args = args


        self.sisnr = SI_SNR()
        self.sasdr  = SA_SDR()

        self.sdr = SDR()


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
        i=1
        for epoch in range(self.start_epoch, self.args.epochs+1):
            self.joint_loss_weight=epoch
            if self.args.distributed: self.args.train_sampler.set_epoch(epoch)

#             Train
            self.model.train()
            start = time.time()
            tr_loss,tr_sisnr = self._run_one_epoch(data_loader = self.train_data, state='train',epoch=epoch)
            reduced_tr_loss = self._reduce_tensor(tr_loss)
            reduced_tr_sisnr  = self._reduce_tensor(tr_sisnr)

            if self.print: print('Train Summary | End of Epoch {0} | Time {1:.2f}s | Current time {2} |'
                      'Train Loss {3:.3f}| SI_SNR {4:.3f}'.format(
                        epoch, time.time() - start,datetime.now(),reduced_tr_loss,reduced_tr_sisnr))

            #Validation
            self.model.eval()
            start = time.time()
            with torch.no_grad():
                val_loss,val_sisnr = self._run_one_epoch(data_loader = self.validation_data, state='val')
                reduced_val_loss = self._reduce_tensor(val_loss)
                reduced_val_sisnr  = self._reduce_tensor(val_sisnr)
                if self.print: print('Valid Summary | End of Epoch {0} | Time {1:.2f}s | Current time {2} |'
                        'Valid Loss {3:.3f}| SI_SNR {4:.3f}'.format(
                            epoch, time.time() - start, datetime.now(),reduced_val_loss,reduced_val_sisnr))
            
            # Check whether to adjust learning rate and early stop
            find_best_model = False
            if (reduced_val_loss )>= self.best_val_loss:
                self.val_no_impv += 1
                if self.val_no_impv >= 10:
                    if self.print: print("No imporvement for 10 epochs, early stopping.")
                    break
            else:
                self.val_no_impv = 0
                self.best_val_loss = reduced_val_loss
                find_best_model=True

            if self.val_no_impv % 3==0 and self.val_no_impv>0:
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
                    self.writer.add_scalar('Validation_sisnr', reduced_val_sisnr, epoch)
                    self.writer.add_scalar('Train_sisnr', reduced_tr_sisnr, epoch)

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
        total_sisnr=0
        self.accu_count = 0
        
        self.optimizer.zero_grad()
        for i, (a_mix,a_mix_mfcc,a_tgt,v_tgt) in enumerate(data_loader):
            a_mix = a_mix.cuda().squeeze(0).float()
            a_tgt = a_tgt.cuda().squeeze(0).float()
            v_tgt = v_tgt.cuda().squeeze(0).float()
            a_mix_mfcc = a_mix_mfcc.cuda().squeeze(0).float()

            with autocast():

                est_a_tgt = self.model(a_mix,a_mix_mfcc,v_tgt)
                
                sisnr = self.sisnr(est_a_tgt,a_tgt)
                sdr_loss = -self.sdr(est_a_tgt,a_tgt)
                loss = sdr_loss 


            if state =='train':
                
                self.accu_count += 1
                
                step+=1
                total_loss+=loss.data
                total_sisnr+=sisnr.data

                if self.args.accu_grad:
                    loss = loss/(self.args.effec_batch_size / self.args.batch_size)
                    scaler.scale(loss).backward()
                    if self.accu_count == (self.args.effec_batch_size / self.args.batch_size):
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_norm, norm_type=2)
                        scaler.step(self.optimizer)
                        scaler.update()
                        self.optimizer.zero_grad()
                        self.accu_count = 0
                else:

                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_norm, norm_type=2)
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad()
                if i%1000==0:
                    print('step:{}/{} avg loss:{:.3f},sisnr:{:.3f}'.format(step, total_step,total_loss / (i+1),total_sisnr/(i+1)))
                
            else: 
                step+=1
                total_loss+=loss.data
                total_sisnr+=sisnr.data

        return total_loss / (i+1), total_sisnr/(i+1)

    def _reduce_tensor(self, tensor):
        if not self.args.distributed: return tensor
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= self.args.world_size
        return rt
