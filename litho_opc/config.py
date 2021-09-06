import argparse
import os
from functools import partial
import time
import random
import string


def get_parser(name):
    """ make default formatted parser """
    parser = argparse.ArgumentParser(name, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # print default value always
    parser.add_argument = partial(parser.add_argument, help=' ')
    return parser


def parse_gpus(gpus):
    if gpus == 'all':
        return list(range(torch.cuda.device_count()))
    else:
        return [int(s) for s in gpus.split(',')]


class BaseConfig(argparse.Namespace):
    def print_params(self, prtf=print):
        prtf("")
        prtf("Parameters:")
        for attr, value in sorted(vars(self).items()):
            prtf("{}={}".format(attr.upper(), value))
        prtf("")

    def as_markdown(self):
        """ Return configs as markdown format """
        text = "|name|value|  \n|-|-|  \n"
        for attr, value in sorted(vars(self).items()):
            text += "|{}|{}|  \n".format(attr, value)

        return text


class OPC_Config(BaseConfig):
    def build_parser(self):
        parser = get_parser("OPC config")
        parser.add_argument('--input_file', required=True)
        parser.add_argument('--opc', type=int, required=True)
        parser.add_argument('--gpu_ind', type=int, default=0, required=True)
        parser.add_argument('--alpha', type=float, default=50, )
        parser.add_argument('--beta', type=float, default=50, )
        parser.add_argument('--step_size', type=float, default=5e6, )
        parser.add_argument('--init_bound', type=float, default=1, )
        parser.add_argument('--init_bias', type=float, default=0.5, )
        parser.add_argument('--momentum', type=float, default=0.05, )
        parser.add_argument('--frac', type=float, default=0, required=False)
        parser.add_argument('--ratio', type=float, default=1, )
        parser.add_argument('--max_step', type=int, default=200)

        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))
