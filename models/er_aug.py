import torch
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from utils.pcr_transforms_aug import transforms_aug


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='ER with strong (SupCon-style) augmentation.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser


class ErAug(ContinualModel):
    NAME = 'er_aug'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(ErAug, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)

    def observe(self, inputs, labels, not_aug_inputs, index_):

        real_batch_size = inputs.shape[0]

        # --- Current-task batch: original + strongly augmented view ---
        batch_x = inputs.to(self.device)
        batch_y = labels.to(self.device)
        batch_x_aug = torch.stack([transforms_aug[self.args.dataset](batch_x[idx].cpu())
                                   for idx in range(batch_x.size(0))]).to(self.device)
        cat_inputs = torch.cat((batch_x, batch_x_aug))
        cat_labels = torch.cat((batch_y, batch_y))

        # --- Buffer batch: original + strongly augmented view ---
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_inputs = buf_inputs.to(self.device)
            buf_labels = buf_labels.to(self.device)
            buf_inputs_aug = torch.stack([transforms_aug[self.args.dataset](buf_inputs[idx].cpu())
                                          for idx in range(buf_inputs.size(0))]).to(self.device)
            cat_inputs = torch.cat((cat_inputs, buf_inputs, buf_inputs_aug))
            cat_labels = torch.cat((cat_labels, buf_labels, buf_labels))

        self.opt.zero_grad()
        outputs = self.net(cat_inputs)
        loss = self.loss(outputs, cat_labels)
        loss.backward()
        self.opt.step()

        # Reservoir update with the (non-augmented) current-task samples.
        self.buffer.add_data(examples=not_aug_inputs[:real_batch_size],
                             labels=labels[:real_batch_size])

        return loss.item()
