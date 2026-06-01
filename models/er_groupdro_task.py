import torch
import torch.nn.functional as F
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='ER with GroupDRO-style robust task weighting.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--gdro_eta', type=float, default=0.01,
                        help='GroupDRO group-weight step size (eta).')
    parser.add_argument('--n_classes_per_task', type=int, default=10,
                        help='Number of classes per task (e.g. 10 for Split CIFAR-100, '
                             '20 for Split Mini-/Tiny-ImageNet). Used to map labels to '
                             'task groups for GroupDRO.')
    return parser



class Ergroupdrotask(ContinualModel):
    NAME = 'er_groupdro_task'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(Ergroupdrotask, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.n_classes_per_task = int(self.args.n_classes_per_task)
        # Total classes from the backbone classifier; q has one weight per task.
        self.total_classes = int(self.net.num_classes)
        self.n_tasks = self.total_classes // self.n_classes_per_task
        self.log_q = torch.zeros(self.n_tasks, device=self.device)

    def observe(self, inputs, labels, not_aug_inputs, index_):

        real_batch_size = inputs.shape[0]

        self.opt.zero_grad()

        # Combine current + replay batches (buffer samples transformed on retrieval).
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

        outputs = self.net(cat_inputs)
        # Per-sample cross-entropy so it can be grouped by task.
        per_sample_loss = F.cross_entropy(outputs, cat_labels, reduction='none')

        # Group (task) id for each sample from its label (disjoint class sets).
        group_ids = torch.div(cat_labels, self.n_classes_per_task,
                              rounding_mode='floor').long()
        present_groups = torch.unique(group_ids)

        # Mean loss per present group.
        group_losses = torch.stack([per_sample_loss[group_ids == g].mean()
                                    for g in present_groups])

        # --- GroupDRO log-additive weight update (no grad, present groups only) ---
        with torch.no_grad():
            self.log_q[present_groups] += self.args.gdro_eta * group_losses.detach()


        w_present = torch.softmax(self.log_q[present_groups], dim=0)
        loss = (w_present * group_losses).sum()

        loss.backward()
        self.opt.step()

        # Reservoir update with the (non-augmented) current-task samples.
        self.buffer.add_data(examples=not_aug_inputs[:real_batch_size],
                             labels=labels[:real_batch_size])

        return loss.item()
