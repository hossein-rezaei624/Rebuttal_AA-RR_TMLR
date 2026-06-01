import numpy as np
import torch
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Experience Replay with Mixup.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--mixup_alpha', type=float, default=1.0,
                        help='Beta distribution parameter for Mixup.')
    return parser


def mixup_data(x, alpha=1.0):
    """Returns mixed inputs, the permutation index, and the lambda."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, index, lam


class ErMixup(ContinualModel):
    NAME = 'er_mixup'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(ErMixup, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)

    def observe(self, inputs, labels, not_aug_inputs, index_):

        real_batch_size = inputs.shape[0]

        self.opt.zero_grad()

        # Combine current and replay batches (buffer samples are transformed on retrieval).
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            cat_inputs = torch.cat((inputs, buf_inputs))
            cat_labels = torch.cat((labels, buf_labels))
        else:
            cat_inputs = inputs
            cat_labels = labels

        cat_inputs = cat_inputs.to(self.device)
        cat_labels = cat_labels.to(self.device)

        # Apply Mixup on the combined batch, then use mixed (soft-label) CE.
        mixed_inputs, perm_index, lam = mixup_data(cat_inputs, self.args.mixup_alpha)
        outputs = self.net(mixed_inputs)
        labels_a, labels_b = cat_labels, cat_labels[perm_index]
        loss = lam * self.loss(outputs, labels_a) + (1 - lam) * self.loss(outputs, labels_b)

        loss.backward()
        self.opt.step()

        # Reservoir update with the (non-mixed, non-augmented) current-task samples.
        self.buffer.add_data(examples=not_aug_inputs[:real_batch_size],
                             labels=labels[:real_batch_size])

        return loss.item()
