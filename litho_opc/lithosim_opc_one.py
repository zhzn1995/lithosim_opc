import os
import numpy as np
from joblib import Parallel, delayed
import time
import sys
import cv2
import matplotlib.pyplot as plt
import torch
import torch.fft as fft
from config import OPC_Config

config = OPC_Config()
os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_ind)

def read_kernel(kernel_file = 'kernel.txt', scale_file = 'scales.txt'):
    with open(kernel_file) as f:
        data = f.readlines()
    data = [list(map(float, line.split())) for line in data]
    data = [[line[i]+line[i+1]*1j for i in range(0,69,2)] for line in data]
    kernels = np.array([np.array(data[35*i:35*(i+1)]).T for i in range(24)])
    
    with open(scale_file) as f:
        scales = f.readlines()[1:]
    scales = np.array(list(map(float, scales)))
    return kernels, scales

def shift_kernel(kernel, size=2048, ):
    k = np.zeros((size,size), dtype=np.complex64)
    k[:18, :18] = kernel[17:, 17:]
    k[:18, -17:] = kernel[-18:, :17]
    k[-17:, :18] = kernel[:17, -18:]
    k[-17:, -17:] = kernel[:17, :17]
    return k

def litho_forward(img, kernels, scales, thr=0.225, beta=50):
    img_fft = fft.fft2(img)
    res = torch.abs(fft.ifft2(torch.mul(kernels, img_fft)))
    res = torch.mul(scales.unsqueeze(1).unsqueeze(2), torch.square(res))
    res = torch.sum(res, 0)
    return torch.sigmoid(beta*(res-thr))

def litho_show(img, kernels, scales, thr=0.225):
    img_fft = fft.fft2(torch.heaviside(img-0.5, torch.tensor([1.], dtype=float).to('cuda')))
    img_fft = fft.fft2(img)
    res = torch.abs(fft.ifft2(torch.mul(kernels, img_fft)))
    res = torch.mul(scales.unsqueeze(1).unsqueeze(2), torch.square(res))
    res = torch.sum(res, 0)
    res[res>=thr] = 1
    res[res<thr] = 0
    return res.cpu().detach().numpy()

def loss_func(litho_out, target):
    return torch.sum(torch.square(litho_out-target))

def read_patterns(file_name, ratio=1):
    patterns = []
    with open(file_name) as f:
        for line in f:
            intline = list(map(lambda x:int(x), line.split()))
            if ratio>1:
                intline = list(map(lambda x:int((x-2400)*ratio+2400), intline))
            patterns.append(np.array([[intline[2*i], intline[2*i+1]] for i in range(len(intline)//2)]))
    return patterns

def pattern2segs(pattern):
    segs = []
    for i in range(len(pattern)-1):
        segs.append([pattern[i],pattern[i+1]])
    return np.array(segs)

def seg_type(seg):
    if seg[0,0] == seg[1,0]:
        return 'v'
    elif seg[0,1] == seg[1,1]:
        return 'h'
    else:
        raise Exception('illegal segment!')

def min_dist_of_polys(poly1, poly2):
    min_dist, min_direction, min_smaller = 1e10, None, None
    segs1 = pattern2segs(poly1)
    segs2 = pattern2segs(poly2)
    for seg1 in segs1:
        for seg2 in segs2:
            # print(seg1)
            # print(seg2)
            dir1 = seg_type(seg1)
            dir2 = seg_type(seg2)
            if dir1 == dir2:
                if dir1 == 'v':
                    x1 = seg1[0,0]
                    x2 = seg2[0,0]
                    dist = x1-x2
                    if not (seg1[:,1].min() >= seg2[:,1].max() or seg2[:,1].min() >= seg1[:,1].max()):
                        if abs(dist) < min_dist:
                            min_dist = abs(dist)
                            min_direction = dir1
                            if dist > 0:
                                min_smaller = 1
                            else:
                                min_smaller = 0
                else:
                    y1 = seg1[0,1]
                    y2 = seg2[0,1]
                    dist = y1-y2
                    if not (seg1[:,0].min() >= seg2[:,0].max() or seg2[:,0].min() >= seg1[:,0].max()):
                        if abs(dist) < min_dist:
                            min_dist = abs(dist)
                            min_direction = dir1
                            if dist > 0:
                                min_smaller = 1
                            else:
                                min_smaller = 0
    return min_dist, min_direction, min_smaller

def move_poly(patterns, frac=0.25, hotspot=True):
    min_dist, min_direction, min_smaller = 1e10, None, None
    for i in range(len(patterns)):
        for j in range(i+1, len(patterns)):
            dist, direction, smaller = min_dist_of_polys(patterns[i], patterns[j])
            # print(dist)
            if dist < min_dist:
                move_obj = i
                min_dist = dist
                min_direction = direction
                min_smaller = smaller
    if (hotspot and min_smaller) or (not hotspot and not min_smaller):
        move_dist = frac * min_dist
    else:
        move_dist = -frac * min_dist
    move_dist = int(move_dist)
    if min_direction == 'h':
        patterns[move_obj] += np.array([0, move_dist])
    elif min_direction == 'v':
        patterns[move_obj] += np.array([move_dist, 0])

def check_patterns(patterns, max_illegal=2):
    illegal = 0
    for pattern in patterns:
        segs_length = []
        for i in range(len(pattern)-1):
            segs_length.append(abs((pattern[i]-pattern[i+1]).sum()))
        if max(segs_length)>512 and min(segs_length)<75:
            illegal += 1
    if illegal>max_illegal:
        raise Exception('Input Layout Incompatible!')
        

def generate_one(coord_name, work_path='.', frac=0.25, ratio=1):
    hotspot = coord_name.split('/')[-1].split('_')[1]=='1'
    patterns = read_patterns(os.path.join(work_path, coord_name), ratio=ratio)
    check_patterns(patterns)
    move_poly(patterns, frac, hotspot)
    img = np.zeros([4800,4800], dtype=np.uint8)
    for pattern in patterns:
        cv2.fillPoly(img, [pattern[:-1]], 255)
    img_out = np.zeros([2048,2048], dtype=np.uint8)
    img_out[512:3*512, 512:3*512] = img[1800:2824, 1800:2824]
    return img_out

def lithosim_opc_one():
    alpha = config.alpha
    beta = config.beta
    learning_rate = config.step_size
    init_bound = config.init_bound
    init_bias = config.init_bias
    momentum = config.momentum
    frac = config.frac
    ratio = config.ratio
    txtname = config.input_file
    print(config.opc)
    if config.opc:
        act_type = 'OPC'
    else:
        act_type = 'Lithosim'

    input_img = generate_one(txtname, frac=frac, ratio=ratio)
    input_img[input_img > 0] = 1
    target_img = torch.tensor(input_img, requires_grad=False, dtype=float).to('cuda')

    kernels, scales = read_kernel()
    kernels = np.array([shift_kernel(kernel) for kernel in kernels])
    kernels = torch.tensor(kernels, requires_grad=False).to('cuda')
    scales = torch.tensor(scales, requires_grad=False).to('cuda')
    
    if config.opc:
        mask_img = torch.tensor((input_img-0.5+init_bias)*init_bound, dtype=float).to('cuda')
        mask_img.requires_grad = True
        optimizer = torch.optim.SGD([mask_img], lr=learning_rate, momentum=momentum)
        for i in range(config.max_step):
            # print('*'*30+'Round %d'%i+'*'*30)
            sig_img = torch.sigmoid(alpha*(mask_img-init_bias))
            litho_out = litho_forward(sig_img, kernels, scales, beta=beta)
            err = loss_func(litho_out, target_img)
            # print(err)
            optimizer.zero_grad()
            err.backward()
            # print(mask_img.grad.abs().max())
            # print(mask_img.abs().max())
            optimizer.step()
    else:
        mask_img = torch.tensor((input_img-0.5+init_bias)*init_bound, dtype=float).to('cuda')
        sig_img = torch.sigmoid(alpha*(mask_img-init_bias))
    
    cv2.imwrite('%s_input.png'%(txtname[:-4]), 255*input_img[512-10:3*512+10, 512-10:3*512+10])
    cv2.imwrite('%s_%s.png'%(txtname[:-4], act_type), 255*litho_show(sig_img, kernels, scales)[512-10:3*512+10, 512-10:3*512+10])

if __name__ == '__main__':
    lithosim_opc_one()