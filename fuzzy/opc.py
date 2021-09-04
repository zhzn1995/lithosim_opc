import os
import numpy as np
from joblib import Parallel, delayed
import time
import sys
import cv2
import matplotlib.pyplot as plt
import torch
import torch.fft as fft

os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
img_path = sys.argv[2]
if not os.path.exists(img_path):
    raise Exception('Invalid path: %s not exists!'%img_path)
obj_path = sys.argv[3]
if not os.path.exists(obj_path):
    os.makedirs(obj_path, exist_ok=True)

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
    return torch.heaviside(res-thr, torch.tensor([1.], dtype=float).to('cuda')).cpu().detach().numpy()
    # res[res>=thr] = 255
    # res[res<thr] = 0
    # return res.cpu().detach().numpy()

def loss_func(litho_out, target):
    return torch.sum(torch.square(litho_out-target))


def opc(img_name, kernels, scales):
    print('*'*60)
    print('Input Image Name: %s'%img_name)
    alpha = 50
    learning_rate = 5e6

    input_img = cv2.imread(os.path.join(img_path, img_name))[:,:,0]
    input_img[input_img > 0] = 1
    target_img = torch.tensor(input_img, requires_grad=False, dtype=float).to('cuda')

    mask_img = torch.tensor(input_img, dtype=float).to('cuda')
    mask_img.requires_grad = True
    optimizer = torch.optim.SGD([mask_img], lr=learning_rate, momentum=0.02)
    for i in range(200):
        print('Round %d'%i)
        sig_img = torch.sigmoid(alpha*(mask_img-0.5))
        litho_out = litho_forward(sig_img, kernels, scales)
        err = loss_func(litho_out, target_img)
        print(err)
        lithores = litho_show(sig_img, kernels, scales).astype(np.uint8)[512:3*512, 512:3*512]*255
        print(lithores.max())
        cv2.imwrite(os.path.join(obj_path, img_name[:-4] + '_%d'%i + '.png'), lithores)
        # plt.figure()
        # plt.imshow(sig_img.cpu().detach().numpy())
        # plt.show()
        # plt.figure()
        # plt.imshow(litho_show(sig_img, kernels, scales))
        # plt.show()
        optimizer.zero_grad()
        err.backward()
        print(mask_img.grad.abs().max())
        print(mask_img.abs().max())
        optimizer.step()

if __name__ == '__main__':
    kernels, scales = read_kernel()
    kernels = np.array([shift_kernel(kernel) for kernel in kernels])

    kernels = torch.tensor(kernels, requires_grad=False).to('cuda')
    scales = torch.tensor(scales, requires_grad=False).to('cuda')

    imgs = os.listdir(img_path)
    for img in imgs:
        if img[-4:] == '.png':
            opc(img, kernels, scales)