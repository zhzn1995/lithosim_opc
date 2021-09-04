import os
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
from common import find_mxnet, data, fit
from common.util import download_file
import mxnet as mx

def add_binary_args(parser):
    parser.add_argument('--bits-w', type=int, default=1,
                       help='number of bits for weights')
    parser.add_argument('--bits-a', type=int, default=1,
                       help='number of bits for activations')

if __name__ == '__main__':
    # download data
    train_fname, val_fname = 'data/128x128/train.rec','data/128x128/val.rec'

    # parse args
    parser = argparse.ArgumentParser(description="train cifar10",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    fit.add_fit_args(parser)
    data.add_data_args(parser)
    data.add_data_aug_args(parser)
    data.set_data_aug_level(parser, 3)
    parser.add_argument('--pretrained', type=str,
                help='the pre-trained model')
    add_binary_args(parser)

    parser.set_defaults(
        gpus           = '0',
        # network
        network        = 'resnet-binary',
        num_layers     = 18,
        # data
        data_train     = train_fname,
        data_val       = val_fname,
        num_classes    = 2,
        num_examples   = 18669,
        image_shape    = '1,128,128',
        # train
        batch_size     = 256,
        num_epochs     = 300,
        lr_step_epochs = '50,125,250',
        optimizer      = 'Nadam',
        disp_batches   = 50,
        lr             = 0.015,
        lr_facto       = 0.15,
        model_prefix   = 'ckpt/hotspot',
        # load_epoch     = 139,
    )

    parser.add_argument('--log', dest='log_file', type=str, default="train.log",
                    help='save training log to file')
    args = parser.parse_args()

    # set up logger    
    log_file = args.log_file
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if log_file:
        fh = logging.FileHandler(log_file)
        logger.addHandler(fh)

    # load network
    from importlib import import_module
    net = import_module('symbols.'+args.network)
    sym = net.get_symbol(**vars(args))

    devs = mx.cpu() if args.gpus is None or args.gpus is '' else [
    mx.gpu(int(i)) for i in args.gpus.split(',')]
    
    #load pretrained
    args_params=None
    auxs_params=None
    
    # train
    if args_params and auxs_params:
        fit.fit(
            args, 
            sym, 
            data.prepair_data_hotspot, 
            arg_params=args_params, 
            aux_params=auxs_params)
    else:
        fit.fit(
            args, 
            sym, 
            data.prepair_data_hotspot)
