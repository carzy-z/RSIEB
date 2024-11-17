import torch
import torch.nn as nn
import torch.nn.utils as utils
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

import os, sys, time
from telnetlib import IP
import argparse
import numpy as np
from tqdm import tqdm

from tensorboardX import SummaryWriter

from utils import post_process_depth, flip_lr, silog_loss, compute_errors, eval_metrics, entropy_loss, colormap, \
    block_print, enable_print, normalize_result, inv_normalize, convert_arg_line_to_args, colormap_magma
from networks.NewCRFDepth import NewCRFDepth
from networks.depth_update import *
from datetime import datetime
from sum_depth import Sum_depth

from dataloaders.levir_dataloader import NewDataLoader

parser = argparse.ArgumentParser(description='IEBins PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--mode', type=str, help='train or test', default='train')
parser.add_argument('--model_name', type=str, help='model name', default='iebins')

# Dataset
parser.add_argument('--data_path', type=str, help='path to the data', default="/data1/zhouhongwei/depth_datasets")
parser.add_argument('--gt_path', type=str, help='path to the groundtruth data', default="/data1/zhouhongwei/depth_datasets")
parser.add_argument('--filenames_file', type=str, help='path to the filenames text file', default="/data1/zhouhongwei/IEBins/uav_data_splits/LEVIR_med_testing_list.txt")
parser.add_argument('--input_height', type=int, help='input height', default=480)
parser.add_argument('--input_width', type=int, help='input width', default=640)
parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=10)
parser.add_argument('--min_depth', type=float, help='minimum depth in estimation', default=0.1)
parser.add_argument('--distributed', type=bool, help='NONE', default=False)

parser.add_argument('--batch_size', type=int, help='batch size', default=4)
parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=1)




if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()


dataloader = NewDataLoader(args, 'train')


if __name__ == '__main__':
    for step, eval_sample_batched in enumerate(dataloader.data):
        if step==1:
            break
