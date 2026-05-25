import torch
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from utils.pcr_loss import SupConLoss
from utils.pcr_transforms_aug import transforms_aug

import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def update_distribution(dist, current_data):
    
    for data in current_data:
        temp_label = int(data)
        dist[int(temp_label)] += 1
    
    # print('dist vec: ',dist)
    return dist

class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, cls_num_list, total_tasks, class_size, which_task):
        super().__init__()
        # numerator = (class_size / total_tasks) * (which_task+1)
        cls_prior = cls_num_list / sum(cls_num_list)
        cls_prior = torch.FloatTensor(cls_prior).cuda()
        self.log_prior = torch.log(cls_prior).unsqueeze(0)

    def forward(self, logits, labels):
        adjusted_logits = logits + self.log_prior
        # print('min and max target balanced softmax: ', labels.min(), labels.max())
        label_loss = F.cross_entropy(adjusted_logits, labels)

        return label_loss


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='DELTA: Decoupling Long-Tailed Online Continual Learning.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    
    return parser


class Delta(ContinualModel):
    NAME = 'delta'
    COMPATIBILITY = ['class-il']

    def __init__(self, backbone, loss, args, transform):
        super(Delta, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.task = 0
        self.epoch = 0
        self.class_size = None
        self.tasks = None

    def begin_train(self, dataset):
        self.class_size = dataset.N_TASKS * dataset.N_CLASSES_PER_TASK
        self.tasks = dataset.N_TASKS
    
    def begin_task(self, dataset):
        self.epoch = 0
        self.task += 1
    
    def end_epoch(self, dataset):
        self.epoch += 1
        

    def observe(self, inputs, labels, not_aug_inputs, index_):
        
        real_batch_size = inputs.shape[0]

        distribution_vector = np.zeros(self.class_size, dtype='float')
        loss_func = None

        batch_x, batch_y = inputs, labels
      
        batch_x = batch_x.to(self.device)      
        batch_y = batch_y.to(self.device)

        loss = torch.tensor(0.0)

        #Stage 1
        if not self.buffer.is_empty():
            mem_x, mem_y = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)

            for param in self.net._features.parameters():
                param.requires_grad = True

            mem_x = mem_x.to(self.device)          
            mem_y = mem_y.to(self.device)
          
            distribution_vector = update_distribution(distribution_vector, torch.cat((batch_y, mem_y)))
            loss_func = BalancedSoftmaxLoss(np.array(distribution_vector), self.tasks, self.class_size, (self.task - 1)).cuda()

            combined_batch = torch.cat((mem_x, batch_x))
            combined_labels = torch.cat((mem_y, batch_y))
            combined_batch_aug = torch.stack([transforms_aug[self.args.dataset](combined_batch[idx].cpu()) 
                                              for idx in range(combined_batch.size(0))])
            combined_batch_aug = combined_batch_aug.to(self.device)
          
            features = torch.cat([self.net.deltaForward(combined_batch).unsqueeze(1), self.net.deltaForward(combined_batch_aug).unsqueeze(1)], dim=1)
            PSC = SupConLoss(temperature=0.09, contrast_mode='all')
            loss_stage1 = PSC(features, combined_labels)
            loss = loss_stage1
            self.opt.zero_grad()
            loss_stage1.backward()
            self.opt.step()

            #stage 2
            
            for param in self.net._features.parameters():
                param.requires_grad = False

            out = self.net.deltaLogits(combined_batch)
            loss_stage2 = loss_func(out, combined_labels)
            loss += loss_stage2
            self.opt.zero_grad()
            loss_stage2.backward()
            self.opt.step()


        self.buffer.add_data(examples=not_aug_inputs[:real_batch_size],
                             labels=labels[:real_batch_size])
        
        return loss.item()
