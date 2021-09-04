""" Training augmented model """
import os
import torch
import torch.nn as nn
import numpy as np
import time
from tensorboardX import SummaryWriter
from config import AugmentConfig
import utils
from models.augment_cnn import AugmentCNN
from torchsampler import ImbalancedDatasetSampler


config = AugmentConfig()

device = torch.device("cuda")

# tensorboard
writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
writer.add_text('config', config.as_markdown(), 0)

logger = utils.get_logger(os.path.join(config.path, "{}.log".format(config.name)))
config.print_params(logger.info)


def main():
    logger.info("Logger is set - training start")

    # set default gpu device id
    torch.cuda.set_device(config.gpus[0])

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    torch.backends.cudnn.benchmark = True

    # get data with meta info
    input_size, input_channels, n_classes, train_data, valid_data = utils.get_data(
        config.dataset, config.data_path, cutout_length=config.cutout_length, validation=True, val_path=config.val_set, fealen=config.fealen, inputsize=config.input_size, aug=True)

    criterion = nn.CrossEntropyLoss().to(device)
    # def cross_entropy(pred, soft_targets):
    #     logsoftmax = nn.LogSoftmax(dim=1)
    #     return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1)).to(device)
    # criterion = cross_entropy
    
    use_aux = config.aux_weight > 0.
    model = AugmentCNN(input_size, input_channels, config.init_channels, n_classes, config.layers,
                       use_aux, config.genotype)
    model = nn.DataParallel(model, device_ids=config.gpus).to(device)

    # model size
    mb_params = utils.param_size(model)
    logger.info("Model size = {:.3f} MB".format(mb_params))

    # weights optimizer
    optimizer = torch.optim.SGD(model.parameters(), config.lr, momentum=config.momentum,
                                weight_decay=config.weight_decay)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config.batch_size,
                                            #    shuffle=True,
                                               sampler=ImbalancedDatasetSampler(train_data),
                                               num_workers=config.workers,
                                               pin_memory=True)
    # train_loader = torch.utils.data.DataLoader(valid_data,
    #                                            batch_size=config.batch_size,
    #                                            sampler=ImbalancedDatasetSampler(valid_data),
    #                                         #    shuffle=True,
    #                                            num_workers=config.workers,
    #                                            pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_data,
                                               batch_size=config.batch_size,
                                               shuffle=False,
                                               num_workers=config.workers,
                                               pin_memory=True)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs)

    best_top1 = 0.
    # training loop
    for epoch in range(config.epochs):
        lr_scheduler.step()
        drop_prob = config.drop_path_prob * epoch / config.epochs
        model.module.drop_path_prob(drop_prob)

        # training
        train(train_loader, model, optimizer, criterion, epoch)

        # validation
        cur_step = (epoch+1) * len(train_loader)
        top1 = validate(valid_loader, model, criterion, epoch, cur_step)

        # save
        if best_top1 < top1:
            best_top1 = top1
            is_best = True
        else:
            is_best = False
        utils.save_checkpoint(model, config.path, is_best, epoch)

        print("")

    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))


def train(train_loader, model, optimizer, criterion, epoch):
    confusion = utils.SumMeter()
    losses = utils.AverageMeter()

    cur_step = epoch*len(train_loader)
    cur_lr = optimizer.param_groups[0]['lr']
    logger.info("Epoch {} LR {}".format(epoch, cur_lr))
    writer.add_scalar('train/lr', cur_lr, cur_step)

    model.train()

    for step, (X, y) in enumerate(train_loader):
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        N = X.size(0)

        optimizer.zero_grad()
        logits, aux_logits = model(X)
        loss = criterion(logits, y)
        if config.aux_weight > 0.:
            loss += config.aux_weight * criterion(aux_logits, y)
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        conf_mat = utils.binary_conf_mat(logits, y)
        losses.update(loss.item(), N)
        confusion.update(conf_mat)
        if conf_mat[1, 0] + conf_mat[1, 1] > 0:
            recall = conf_mat[1, 1] / (conf_mat[1, 0] + conf_mat[1, 1])
        else:
            recall = 0
        fa = conf_mat[0, 1]
        f1_score = utils.binary_f1_score(conf_mat)
        if step % config.print_freq == 0 or step == len(train_loader)-1:
            logger.info(
                "Train: [{:3d}/{}] Step {:03d}/{:03d} Loss: {:.3f} Recall: {:.3f} FA: {:d} F1: {:3f}".format(
                    epoch+1, config.epochs, step, len(train_loader)-1, losses.avg, recall, fa, f1_score))
            logger.info(conf_mat.flatten())

            writer.add_scalar('train/loss', loss.item(), cur_step)
            # writer.add_scalar('train/top1', prec1.item(), cur_step)
            # writer.add_scalar('train/top5', prec5.item(), cur_step)
        cur_step += 1
    
    recall = confusion.val[1,1]/(confusion.val[1,0]+confusion.val[1,1])
    fa = confusion.val[0,1]
    f1_score = utils.binary_f1_score(confusion.val)
    logger.info("Train: [{:3d}/{}] Final Recall {:.4%} FA {:d} F1: {:.4%}".format(epoch+1, config.epochs, recall, fa, f1_score))
    logger.info(confusion.val.flatten())


def validate(valid_loader, model, criterion, epoch, cur_step):
    confusion = utils.SumMeter()
    losses = utils.AverageMeter()

    model.eval()

    with torch.no_grad():
        start = time.time()
        for step, (X, y) in enumerate(valid_loader):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            N = X.size(0)

            logits, _ = model(X)
            loss = criterion(logits, y)

            conf_mat = utils.binary_conf_mat(logits, y)
            losses.update(loss.item(), N)
            confusion.update(conf_mat)
            if conf_mat[1, 0] + conf_mat[1, 1] > 0:
                recall = conf_mat[1, 1] / (conf_mat[1, 0] + conf_mat[1, 1])
            else:
                recall = 0
            fa = conf_mat[0, 1]
            f1_score = utils.binary_f1_score(conf_mat)

            if step % config.print_freq == 0 or step == len(valid_loader)-1:
                logger.info(
                    "Valid: [{:3d}/{}] Step {:03d}/{:03d} Loss: {:.3f} Recall: {:.3f} FA: {:d} F1: {:3f}".format(
                        epoch+1, config.epochs, step, len(valid_loader)-1, losses.avg, recall, fa, f1_score))
                logger.info(conf_mat.flatten())

    val_time = time.time() - start
    recall = confusion.val[1,1]/(confusion.val[1,0]+confusion.val[1,1])
    fa = confusion.val[0,1]
    f1_score = utils.binary_f1_score(confusion.val)

    writer.add_scalar('val/loss', losses.avg, epoch)
    writer.add_scalar('val/recall', recall, epoch)
    writer.add_scalar('val/fa', fa, epoch)
    writer.add_scalar('val/f1_score', f1_score, epoch)
    
    logger.info("Valid: [{:3d}/{}] Final Recall {:.4%} FA {:d} F1: {:.4%}".format(epoch+1, config.epochs, recall, fa, f1_score))
    logger.info(confusion.val.flatten())
    logger.info("Valid: Time Spent: %fs"%val_time)
    logger.info("\n")

    return f1_score


if __name__ == "__main__":
    main()
