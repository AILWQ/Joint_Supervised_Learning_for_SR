import datetime
import math
import os
import random
import time
import numpy as np
import omegaconf
import argparse
from torchinfo import summary
from torch import nn

from loss_function.ContrastiveLoss import ContrastiveLoss
from src.architectures.expression_generator import model
from src.set_dataset_loader import set_loader
from src.utils import count_parameters
from src.utils import epoch_time, plot_losses
import torch.distributed as dist
import torch.utils.data.distributed

start_epoch = 0
now = datetime.datetime.now()
date = now.strftime("%Y%m%d_%H%M%S")
dir_name = ""  # time
log_file = './log/{}'.format(dir_name)
checkpoint_file = './weights/{}'.format(dir_name)
loss_file = log_file + '/loss_' + date + ".txt"
if not os.path.exists(log_file):
    os.mkdir(log_file)
if not os.path.exists(checkpoint_file):
    os.mkdir(checkpoint_file)

# load config
cfg = omegaconf.OmegaConf.load('./config.yaml')

seed = 6
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

train_dataset, val_dataset = set_loader(cfg)

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
parser.add_argument('--world_size', default=4, help="n_gpus")
args = parser.parse_args()
print(args.local_rank)

dist.init_process_group(backend='nccl', rank=args.local_rank, world_size=args.world_size)
torch.cuda.set_device(args.local_rank)

# train model
model = model(cfg=cfg.architecture)
model = model.cuda(args.local_rank)
summary(model)

print("Total params: %.2fM" % (count_parameters(model) / 1e6))

optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

# DDP training
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=False,
                                                  output_device=args.local_rank)

cross_entropy = nn.CrossEntropyLoss(ignore_index=0)
contrastive_learning = ContrastiveLoss(temperature=0.5)

tokens = 0  # counter used for learning rate decay

train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=cfg.batch_size,
                                           sampler=train_sampler)

val_loader = torch.utils.data.DataLoader(val_dataset,
                                         batch_size=cfg.batch_size)

# start training
for epoch in range(start_epoch, cfg.epochs):
    train_sampler.set_epoch(epoch)
    start_time = time.time()

    model.train()
    losses_train = []
    total_train_step = 0
    train_samples_per_epoch = 0

    for batch in train_loader:
        points = batch[0].cuda(non_blocking=True)
        # print("==>DEBUG || points: ", points, points.shape)
        if points.shape[0] == 1:
            continue
        trg = batch[1].cuda(non_blocking=True)
        eq_labels = batch[2].cuda(non_blocking=True).contiguous().view(-1)
        # print("==>DEBUG || eq_labels: ", eq_labels, eq_labels.shape)
        output, trg, enc_output = model(points, trg)
        output = output.permute(1, 0, 2).contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)

        CE_loss = cross_entropy(output, trg)  # cross_entropy loss
        CL_loss = contrastive_learning(enc_output, eq_labels)  # supervised contrastive learning loss
        loss = (1 - cfg.scale_weight) * CE_loss + cfg.scale_weight * CL_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_norm_clip)
        optimizer.step()

        if cfg.lr_decay:
            tokens += (trg >= 0).sum()  # number of tokens processed this step (i.e. label is not -100)
            if tokens < cfg.warmup_tokens:
                # linear warmup
                lr_mult = float(tokens) / float(max(1, cfg.warmup_tokens))
            else:
                # cosine learning rate decay
                progress = float(tokens - cfg.warmup_tokens) / float(
                    max(1, cfg.final_tokens - cfg.warmup_tokens))
                lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
            lr = cfg.lr * lr_mult
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            lr = cfg.lr

        train_samples_per_epoch += int(points.shape[0])

        total_train_step += 1
        if args.local_rank == 0:
            if total_train_step % 100 == 0:
                end_time = time.time()
                hours, mins, secs = epoch_time(start_time, end_time)
                print('==========================================================')
                # print('*** points shape: {}\ttarget shape: {} ***'.format(points.shape, trg.shape))
                print('Epoch: {} [{}/{}] | Time: {}h {}m {}s | lr: {:.9f}'.format(epoch + 1, total_train_step,
                                                                                  len(train_loader), hours, mins, secs,
                                                                                  lr))
                # print('\tTrain Loss: {:.5f} |  Contrastive Loss: {:.5f}'.format(loss.item(), CL_loss.item()))
                print('==========================================================\n')

        losses_train.append(loss.item())

    print('~~~~~~~~~~~~~~~~{} training expressions of epoch {}~~~~~~~~~~~~~~~~'.format(train_samples_per_epoch, epoch))

    train_loss = float(np.mean(losses_train))

    if args.local_rank == 0:
        losses_val = []
        with torch.no_grad():
            model.eval()
            for batch in val_loader:
                points = batch[0].cuda(non_blocking=True)
                if points.shape[0] == 1:
                    continue
                trg = batch[1].cuda(non_blocking=True)
                output, trg, _ = model(points, trg)
                output = output.permute(1, 0, 2).contiguous().view(-1, output.shape[-1])
                trg = trg[:, 1:].contiguous().view(-1)
                loss = CE_loss(output, trg)
                losses_val.append(loss.item())

            val_loss = float(np.mean(losses_val))

        checkpoint = {
            'epoch': epoch,
            'state_dict': model.module.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        torch.save(checkpoint, '{}/epoch-{:02d}_train_loss-{:.4f}.pth'.format(checkpoint_file, epoch + 1, train_loss))

        open(loss_file, 'a').write('{:.4f} {:.4f}\n'.format(train_loss, val_loss))

        print('Epoch: {} | {}'.format(epoch + 1, 'Val Loss Per Epoch'))
        print('\t Val. Loss: {:.5f} |  Val. PPL: {:.5f}'.format(val_loss, math.exp(val_loss)))

plot_losses(loss_file=loss_file,
            plot_file='{}/loss_'.format(log_file) + date + '.png',
            xlimits=None, ylimits=None)
