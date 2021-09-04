import os
import torch
import torch.nn as nn
import numpy as np
import time
import math
from tensorboardX import SummaryWriter
from config import AugmentConfig
import utils
from models.augment_cnn import AugmentCNN
from torchsampler import ImbalancedDatasetSampler
from geneticalgorithm import geneticalgorithm as ga
import genotypes as gt
import generate_random_arch
import GPy
from platypus.problems import Problem
from platypus.algorithms import NSGAII
from platypus.types import Real, Integer

config = AugmentConfig()

device = torch.device("cuda")

# tensorboard
writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
# writer.add_text('config', config.as_markdown(), 0)

logger = utils.get_logger(os.path.join(config.path, "{}.log".format(config.name)))
config.print_params(logger.info)


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
    config.dataset, config.data_path, cutout_length=config.cutout_length, validation=True, val_path=config.val_set, fealen=config.fealen, inputsize=config.input_size)

criterion = nn.CrossEntropyLoss().to(device)

use_aux = config.aux_weight > 0.

# weights optimizer


train_loader = torch.utils.data.DataLoader(train_data,
                                            batch_size=config.batch_size,
                                        #    shuffle=True,
                                            sampler=ImbalancedDatasetSampler(train_data),
                                            num_workers=config.workers,
                                            pin_memory=True)
valid_loader = torch.utils.data.DataLoader(valid_data,
                                            batch_size=config.batch_size,
                                            shuffle=True,
                                            num_workers=config.workers,
                                            pin_memory=True)
valid_iterator = iter(valid_loader)

def l2genotype(l, num_node):
    PRIMITIVES = [
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect', # identity
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
    'none'
    ]
    genotype = 'Genotype(normal=['
    for i in range(num_node):
        genotype += '[(\'%s\', %d), (\'%s\', %d)],'%(PRIMITIVES[l[i*4]], l[i*4+1], PRIMITIVES[l[i*4+2]], l[i*4+3])
    genotype += '], normal_concat=range(2, %d), reduce=['%(num_node + 2)
    for i in range(num_node, num_node*2):
        genotype += '[(\'%s\', %d), (\'%s\', %d)],'%(PRIMITIVES[l[i*4]], l[i*4+1], PRIMITIVES[l[i*4+2]], l[i*4+3])
    genotype += '], reduce_concat=range(2, %d))'%(num_node + 2)
    return genotype

def l2graph(l, num_node):
    G_m = np.zeros([num_node+2, num_node+2])
    C_m = np.zeros([num_node+2, num_node+2])
    C_m[0, 0] = 0.15
    C_m[1,1] = 0.25
    for i in range(num_node):
        G_m[i+2, l[i*4+1]] -= 2**l[i*4]
        G_m[l[i*4+1], i+2] -= 2**l[i*4]
        G_m[i+2, i+2] += 2**l[i*4]
        G_m[l[i*4+1], l[i*4+1]] += 2**l[i*4]

        G_m[i+2, l[i*4+3]] -= 2**l[i*4+2]
        G_m[l[i*4+3], i+2] -= 2**l[i*4+2]
        G_m[i+2, i+2] += 2**l[i*4+2]
        G_m[l[i*4+3], l[i*4+3]] += 2**l[i*4+2]

        C_m[i+2, i+2]  = 1/math.sqrt(2**l[i*4]+2**l[i*4+2])
        
    t_m = np.dot(C_m, G_m)
    A_m = np.dot(t_m, C_m)
    eigvals, eigvecs = np.linalg.eig(A_m)
    idx = eigvals.argsort()
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:,idx]
    temp = eigvecs[:, 1:3]
    # print(G_m)
    # print(C_m)
    # print(A_m)
    # print(eigvals)
    return abs(np.reshape(temp.transpose(), [2*(num_node+2)]))

def gene_dist(l, num_node):
    d1 = l2graph(l[:4*num_node], num_node)
    d2 = l2graph(l[4*num_node:], num_node)
    # print(d1, d2)
    return np.append(d1, d2)

def ga_opt(num_node, val_f):
    problem = Problem(num_node*8, 1)
    problem.function = val_f
    for i in range(2):
        for j in range(num_node):
            for k in range(2):
                problem.types[i*num_node*4+j*4+k*2] = Integer(0,6)
                problem.types[i*num_node*4+j*4+k*2+1] = Integer(0,j+1)
    algorithm = NSGAII(problem, population = 50)
    algorithm.run(1000)
    optimized = algorithm.result
    def get_res(res, pro):
        tmp = []
        for i in range(pro.nvars):
            tmp.append(pro.types[i].decode(res.variables[i]))
        return np.array(tmp), res.objectives[0]
    idx = np.random.permutation(len(optimized))[0]
    return get_res(optimized[idx], problem)
    # bound = []
    # for i in range(num_node):
    #     bound += [[0,6], [0,i+1], [0,6], [0,i+1]]
    # bound *= 2
    # mod = ga(function=val_f,dimension=num_node*8,variable_type='int',variable_boundaries=np.array(bound))
    # mod.run()
    # return mod.output_dict['variable']

def train_val(epochs, genotype, val_batches):
    print(genotype)
    model = AugmentCNN(input_size, input_channels, config.init_channels, n_classes, config.layers,
                       use_aux, genotype)
    model = nn.DataParallel(model, device_ids=config.gpus).to(device)

    # model size
    mb_params = utils.param_size(model)
    logger.info("Model size = {:.3f} MB".format(mb_params))

    optimizer = torch.optim.SGD(model.parameters(), config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200)

    scores = []
    for epoch in range(epochs):
        
        drop_prob = config.drop_path_prob * epoch / config.epochs
        model.module.drop_path_prob(drop_prob)

        # training
        train(train_loader, model, optimizer, criterion, epoch)
        lr_scheduler.step()

        # validation
        cur_step = (epoch+1) * len(train_loader)
        top1 = validate(valid_loader, model, criterion, epoch, cur_step, val_batches)
        scores.append(top1)
    logger.info(scores)
    return 1-np.mean(scores)

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

            # writer.add_scalar('train/loss', loss.item(), cur_step)
            # writer.add_scalar('train/top1', prec1.item(), cur_step)
            # writer.add_scalar('train/top5', prec5.item(), cur_step)
        cur_step += 1
    
    recall = confusion.val[1,1]/(confusion.val[1,0]+confusion.val[1,1])
    fa = confusion.val[0,1]
    f1_score = utils.binary_f1_score(confusion.val)
    logger.info("Train: [{:3d}/{}] Final Recall {:.4%} FA {:d} F1: {:.4%}".format(epoch+1, config.epochs, recall, fa, f1_score))
    logger.info(confusion.val.flatten())

def validate(valid_loader, model, criterion, epoch, cur_step, val_batches):
    global valid_iterator
    confusion = utils.SumMeter()
    losses = utils.AverageMeter()

    model.eval()

    with torch.no_grad():
        start = time.time()
        
        for step in range(val_batches):     
            try:
                X, y = next(valid_iterator)
            except StopIteration:
                valid_iterator = iter(valid_loader)
                X, y = next(valid_iterator)

        # for step, (X, y) in enumerate(valid_loader):
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

    # writer.add_scalar('val/loss', losses.avg, epoch)
    # writer.add_scalar('val/recall', recall, epoch)
    # writer.add_scalar('val/fa', fa, epoch)
    # writer.add_scalar('val/f1_score', f1_score, epoch)
    
    logger.info("Valid: [{:3d}/{}] Final Recall {:.4%} FA {:d} F1: {:.4%}".format(epoch+1, config.epochs, recall, fa, f1_score))
    logger.info(confusion.val.flatten())
    logger.info("Valid: Time Spent: %fs"%val_time)
    # logger.info("\n")

    return f1_score

def bayes_opt(num_node):
    dim      = 32      # input's dim
    num_init = 4
    val_batches = 30
    score_epochs = 1
    max_eval = 200
    # xs       = generate_random_arch.random_gen(num_init, num_node)
    # ys       = np.zeros(num_init)
    # for i in range(num_init):
    #     ys[i] = train_val(score_epochs, gt.from_str(l2genotype(xs[i], num_node)), val_batches)
    # print(xs, ys)
    
    # xs =   [[0,0,1,1,2,0,6,1,1,0,3,1,1,3,6,1,3,0,5,0,5,2,1,2,6,3,1,0,4,3,1,2],
    #         [0,0,2,0,0,1,6,1,6,1,2,0,0,4,1,3,4,0,4,0,6,0,5,2,3,1,2,1,6,1,0,3],
    #         [6,1,0,1,4,2,2,2,5,0,0,0,4,3,3,0,4,0,0,1,6,2,6,0,2,0,0,1,5,1,0,3],
    #         [4,1,2,1,1,1,3,2,4,1,1,1,0,0,5,2,6,0,4,1,5,2,6,0,0,3,1,0,5,2,4,3]]
    # xs = np.array(xs)
    # ys = [0.56948933,0.59156493, 0.64, 0.61128812]
    # ys = np.array(ys)

    xs = np.loadtxt('testx.csv', delimiter=',').astype(np.int8)
    ys = np.loadtxt('testy.csv', delimiter=',').astype(np.float32)
    for cnt in range(max_eval):
        print(xs)
        print(ys)
        y1 = ys.reshape(len(ys), 1)
        x1 = (xs-3)/2
        gp_m1 = GPy.models.GPRegression(x1, y1, GPy.kern.RBF(input_dim = dim, ARD = True))
        # gp_m1.kern.mean = 0.5
        gp_m1.kern.variance = np.var(y1)
        gp_m1.kern.lengthscale = np.std(x1, 0)
        gp_m1.likelihood.variance = 1e-3 * np.var(y1)
        gp_m1.optimize()

        def lcb(x):
            x = (np.array(x)-3)/2
            py1, ps2_1 = gp_m1.predict(x.reshape(1, dim))
            ps_1       = np.sqrt(ps2_1)
            lcb1       = py1 - 3 * ps_1
            return [lcb1[0, 0]]
        
        new_x, res_lcb = ga_opt(num_node, lcb)
        print(new_x, res_lcb)
        
        
        # ga_result = ga.run(1000, mdenas.rand_arch(50), population = 50, fix_fun = lcb)

        # idx = np.random.permutation(len(ga_result))[:4]
        # new_x = ga_result[idx]
        # new_y = np.zeros(4)
        # for i in range(4):
        #     new_y[i] = mdenas.predict_acc(xs[i], thread = 4)
        
        py1, ps2_1 = gp_m1.predict((new_x.reshape(1, dim)-3)/2)
        
        print(cnt)
        print('-'*60)
        print(py1, ps2_1)
        new_y = train_val(score_epochs, gt.from_str(l2genotype(new_x, num_node)), val_batches)
        print(new_y)
        ys = np.append(ys, new_y)
        xs = np.concatenate((xs, new_x.reshape(1, dim)), axis=0)
        np.savetxt('testx.csv', xs, fmt='%d', delimiter=',')
        np.savetxt('testy.csv', ys, fmt='%f', delimiter=',')

bayes_opt(4)
# genotype = "Genotype(normal=[[('sep_conv_3x3', 0), ('dil_conv_3x3', 0)], [('sep_conv_3x3', 2), ('dil_conv_5x5', 1)], [('sep_conv_3x3', 1), ('sep_conv_5x5', 2)], [('sep_conv_5x5', 4), ('sep_conv_5x5', 1)]], normal_concat=range(2, 6), reduce=[[('sep_conv_5x5', 1), ('sep_conv_5x5', 0)], [('sep_conv_5x5', 2), ('dil_conv_5x5', 1)], [('dil_conv_3x3', 1), ('sep_conv_3x3', 3)], [('sep_conv_3x3', 4), ('sep_conv_5x5', 2)]], reduce_concat=range(2, 6))"

# train_val(3, gt.from_str(genotype), 20)
