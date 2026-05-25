import torch
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from utils.pcr_loss import SupConLoss
from utils.pcr_transforms_aug import transforms_aug

import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class WeightedLoss(nn.Module):
    def __init__(self, temperature=1.0, class_weights=None, label_freq_flag=False):
        """
        Args:
            temperature (float): Temperature scaling factor.
            class_weights (dict or torch.Tensor, optional): Mapping from class index to weight.
                (If None, all classes are assumed to have weight 1.)
        """
        super(WeightedLoss, self).__init__()
        self.temperature = temperature
        self.class_weights = class_weights
        self.label_freq_flag = label_freq_flag

    def forward(self, anchor_features, labels, proxies):
        """
        Args:
            anchor_features (Tensor): Features of shape (N, d) for N anchor samples.
            labels (Tensor): Ground-truth labels for each anchor sample, shape (N,).
            proxies (Tensor): Proxy vectors (classifier weights) for all classes, shape (C, d).
        Returns:
            loss (Tensor): The computed weighted proxy contrastive loss (scalar).
        """
        # Normalize features and proxies to obtain cosine similarities.
        ##anchor_features = F.normalize(anchor_features, p=2, dim=1, eps=1e-6)
        ##proxies = F.normalize(proxies, p=2, dim=1, eps=1e-6)
        
        # Compute similarity matrix (N x C) and scale by temperature.
        sim_matrix = torch.matmul(anchor_features, proxies.t()) / self.temperature
        
        # Numerical stability: subtract the maximum value per row.
        sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - sim_max.detach()  # logits now is stable
            
        # Compute exp(logits)
        exp_logits = torch.exp(logits)
            
        # Build weight vector (if provided) or use ones.
        if self.class_weights is None:
            weight_vector = torch.ones(proxies.shape[0], device=anchor_features.device, dtype=anchor_features.dtype)
        else:
            if isinstance(self.class_weights, dict):
                weight_list = [self.class_weights.get(c, 0.0) for c in range(proxies.shape[0])]
                weight_vector = torch.tensor(weight_list, device=anchor_features.device, dtype=anchor_features.dtype)
            else:
                weight_vector = self.class_weights.to(anchor_features.device)
        weight_vector = weight_vector.unsqueeze(0)  # Shape: (1, C)

        # Compute frequency of each class in the batch.
        freq = torch.bincount(labels, minlength=proxies.shape[0]).float().to(anchor_features.device)  # Shape: (C,)
        freq = freq.unsqueeze(0)  # Shape: (1, C)
        
        # Multiply frequency by weight_vector to get effective frequency per class.
        if self.label_freq_flag:
            freq_weighted = weight_vector * freq
        else:
            freq_weighted = weight_vector
            
        # Denominator: sum over all classes weighted by frequency.
        denom = torch.sum(exp_logits * freq_weighted, dim=1, keepdim=True)

        # For each anchor, pick the logit corresponding to its true label.
        true_logits = logits.gather(1, labels.view(-1, 1))

        # Numerator (only the true class, also weighted by freq)
        true_class_freq = freq_weighted.squeeze(0).gather(0, labels).unsqueeze(1)  # [N, 1]
        true_logits = true_logits + torch.log(true_class_freq)
        
        # Compute log probability and loss.
        log_prob = true_logits - torch.log(denom)
        loss = -1 * log_prob
        loss = loss.mean()
            
        return loss


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Ours_gamma.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--Power_alpha', type=float, default=1.0,
                        help='Power for first term')
    parser.add_argument('--Power_beta', type=float, default=1.0,
                        help='Power for second term')
    parser.add_argument('--Power_gamma', type=float, default=1.0,
                        help='Power for third term')
    
    return parser


class Oursgamma(ContinualModel):
    NAME = 'ours_gamma'
    COMPATIBILITY = ['class-il']

    def __init__(self, backbone, loss, args, transform):
        super(Oursgamma, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.task = 0
        self.epoch = 0
        self.n_sample_per_task = None
        self.n_classes_ = None
        self.unique_classes = None
        self.class_weights_ = {}

    def begin_train(self, dataset):
        self.n_sample_per_task = dataset.get_examples_number()//dataset.N_TASKS
        self.n_classes_ = dataset.N_CLASSES_PER_TASK * dataset.N_TASKS
    
    def begin_task(self, dataset, train_loader):
        self.epoch = 0
        self.task += 1
        self.unique_classes = set()
        for _, labels, _, _ in train_loader:
            self.unique_classes.update(labels.numpy())
            if len(self.unique_classes)==dataset.N_CLASSES_PER_TASK:
                break
    
    def end_epoch(self, dataset):
        self.epoch += 1
        

    def observe(self, inputs, labels, not_aug_inputs, index_):
        
        real_batch_size = inputs.shape[0]

        if self.task > 1:
            buf_freq = torch.bincount(self.buffer.labels[:self.buffer.num_seen_examples], minlength=self.n_classes_).float().to(self.device)  # Shape: (C,)
        
        self.class_weights_ = {i: (1.0 if i >= ((self.task - 1)*len(self.unique_classes)) 
                                   else (1.0/(self.task - 1))**self.args.Power_alpha * (1.0/(self.task - (i // len(self.unique_classes))))**self.args.Power_beta * 
                                   (buf_freq[i].item()/(buf_freq[list(self.unique_classes)].mean().item() + (self.n_sample_per_task/len(self.unique_classes))))**self.args.Power_gamma) 
                               for i in range(self.task * len(self.unique_classes))}
        
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

            out = self.net(combined_batch, 'features')

            weighted_loss = WeightedLoss(temperature=1.0, class_weights=self.class_weights_)
            loss_stage2 = weighted_loss(out, combined_labels, self.net.linear.weight)
            
            loss += loss_stage2
            self.opt.zero_grad()
            loss_stage2.backward()
            self.opt.step()


        self.buffer.add_data(examples=not_aug_inputs[:real_batch_size],
                             labels=labels[:real_batch_size])
        
        return loss.item()
