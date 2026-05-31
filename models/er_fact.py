import torch
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from utils.pcr_transforms_aug import transforms_aug


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='ER with CIRL-style factorization regularizer.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--fact_lambda', type=float, default=1.0,
                        help='Weight of the factorization regularizer added to the CE loss.')
    return parser


def off_diagonal(x):
    # Flattened view of the off-diagonal elements of a square matrix.
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def factorization_loss(f_a, f_b):
    # Barlow-Twins-style cross-correlation between two views' representations.
    f_a_norm = (f_a - f_a.mean(0)) / (f_a.std(0) + 1e-6)
    f_b_norm = (f_b - f_b.mean(0)) / (f_b.std(0) + 1e-6)
    c = torch.mm(f_a_norm.T, f_b_norm) / f_a_norm.size(0)

    on_diag = torch.diagonal(c).add_(-1).pow_(2).mean()
    off_diag = off_diagonal(c).pow_(2).mean()
    return on_diag + 0.005 * off_diag


class ErFact(ContinualModel):
    NAME = 'er_fact'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(ErFact, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)

    def observe(self, inputs, labels, not_aug_inputs, index_):

        real_batch_size = inputs.shape[0]

        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        # --- Build augmented view of the current batch ---
        cur_x = inputs
        cur_x_aug = torch.stack([transforms_aug[self.args.dataset](cur_x[i].cpu())
                                 for i in range(cur_x.size(0))]).to(self.device)

        # --- Build augmented view of the buffer batch (if any) ---
        have_buffer = not self.buffer.is_empty()
        if have_buffer:
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_inputs = buf_inputs.to(self.device)
            buf_labels = buf_labels.to(self.device)
            buf_x_aug = torch.stack([transforms_aug[self.args.dataset](buf_inputs[i].cpu())
                                     for i in range(buf_inputs.size(0))]).to(self.device)

            # ER-style: concatenate current+buffer (orig and aug) into ONE batch
            # and do ONE forward + ONE CE call over the whole thing. BatchNorm
            # sees the full current+buffer distribution exactly as plain ER does.
            all_inputs = torch.cat([cur_x, buf_inputs, cur_x_aug, buf_x_aug])
            all_labels = torch.cat([labels, buf_labels, labels, buf_labels])

            logits, feats = self.net(all_inputs, returnt='all')

            n_cur, n_buf = cur_x.size(0), buf_inputs.size(0)
            cur_feat_o = feats[:n_cur]
            buf_feat_o = feats[n_cur:n_cur + n_buf]
            cur_feat_a = feats[n_cur + n_buf:2 * n_cur + n_buf]
            buf_feat_a = feats[2 * n_cur + n_buf:]
        else:
            # First batch(es) of task 1 under reservoir: only current orig+aug.
            all_inputs = torch.cat([cur_x, cur_x_aug])
            all_labels = torch.cat([labels, labels])

            logits, feats = self.net(all_inputs, returnt='all')

            n_cur = cur_x.size(0)
            cur_feat_o = feats[:n_cur]
            cur_feat_a = feats[n_cur:]

        # ONE CE call over the entire (orig+aug, current+buffer) batch -- ER-style.
        ce_loss = self.loss(logits, all_labels)

        if have_buffer:
            com_feat_o = torch.cat((cur_feat_o, buf_feat_o))
            com_feat_a = torch.cat((cur_feat_a, buf_feat_a))
            fact = factorization_loss(com_feat_o, com_feat_a)
        else:
            fact = factorization_loss(cur_feat_o, cur_feat_a)
            
        loss = ce_loss + self.args.fact_lambda * fact

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # Reservoir update with the (non-augmented) current-task samples.
        self.buffer.add_data(examples=not_aug_inputs[:real_batch_size],
                             labels=labels[:real_batch_size])

        return loss.item()
