from gdsii.library import Library
from gdsii.elements import *
import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys
def loc(x):
    i=np.array(x)
    return np.array([i[:,0].min(), i[:,1].min(), i[:,0].max(), i[:,1].max()])
def contain(loc1, loc2):
    return loc1[0]<=loc2[0] and loc1[1]<=loc2[1] and loc1[2]>=loc2[2] and loc1[3]>=loc2[3]
def cutpic(dataclass, name):
    if dataclass == "train":
        filename = 'raw/MX_Benchmark%d_clip.gds'%name
    elif dataclass == "val":
        filename = 'raw/array_benchmark%d.gds'%name
    
    with open(filename, 'rb') as stream:
        lib = Library.load(stream)
    alldict = {}
    contents = [i for i in lib[0] if i.layer==10]
    num_contents = len(contents)
    bounds = [i for i in lib[0] if i.layer in range(21, 24)]
    num_bounds = len(bounds)

    print('%s\t%d\t%d'%(filename, num_contents, num_bounds))
    errnum, num = 0, 0
    for content in contents:
        for bound in bounds:
            loctmp=loc(bound.xy)
            # if loctmp[2]-loctmp[0]!=1200 or loctmp[3]-loctmp[1]!=1200 or len(bound.xy)!=5:
            #     continue
            loctmp[0:2]-=1800
            # loctmp[2:4]+=1800
            loctmp[2:4] = loctmp[0:2] + 4800
            basestring = '%d,%d'%(loctmp[0], loctmp[1])
            if contain(loctmp, loc(content.xy)):
                if basestring in alldict:
                    alldict[basestring]+=[content.xy]
                else:
                    alldict[basestring]=[content.xy]
                break
        sys.stdout.write('DONE %d/%d\r'%(num, len(contents)))
        num+=1
    print('DONE')
    nums=[0, 0]
    for bound in bounds:
        loctmp = loc(bound.xy)
        if loctmp[2]-loctmp[0]!=1200 or loctmp[3]-loctmp[1]!=1200 or len(bound.xy)!=5:
            errnum+=1
            # continue
        loctmp[0:2]-=1800
        # loctmp[2:4]+=1800
        loctmp[2:4] = loctmp[0:2] + 4800
        basestring = '%d,%d'%(loctmp[0], loctmp[1])
        img = np.zeros([4800,4800], dtype=np.uint8)
        base = np.array(loctmp[0:2], dtype=np.int64)
        for content in alldict[basestring]:
            pat = np.array(content, dtype=np.int64)
            poly = pat-base
            cv2.fillPoly(img, [poly[:-1]], 255)
        if bound.layer%10 in range(1,3):
            ptype=1
        elif bound.layer%10==3:
            ptype=0
        else:
            print('error!')
            # continue
        img_out = np.zeros([2048,2048], dtype=np.uint8)
        img_out[512:3*512, 512:3*512] = img[1800:2824, 1800:2824]
        cv2.imwrite('img/%s/%d_%d_%d.png'%(dataclass, name, ptype, nums[ptype]) ,img_out)
        nums[ptype] += 1
    print('Bound Errs: %d'%errnum)
    print('Hotspots: %d, Non-hotspots: %d'%(nums[1], nums[0]))

def check_bound_err(dataclass, name):
    if dataclass == "train":
        filename = 'raw/MX_Benchmark%d_clip.gds'%name
    elif dataclass == "val":
        filename = 'raw/array_benchmark%d.gds'%name
    
    with open(filename, 'rb') as stream:
        lib = Library.load(stream)
    alldict = {}
    contents = [i for i in lib[0] if i.layer==10]
    num_contents = len(contents)
    bounds = [i for i in lib[0] if i.layer in range(21, 24)]
    num_bounds = len(bounds)
    # layers = []
    # for i in lib[1]:
    #     if i.layer not in layers:
    #         layers += [i.layer]
    # print(layers)
    print('%s\t%d\t%d'%(filename, num_contents, num_bounds))
    nums = [0, 0]
    errnum = 0
    for bound in bounds:
        if bound.layer%10 in range(1,3):
            ptype=1
        elif bound.layer%10==3:
            ptype=0
        else:
            print('Error Type!')
        loctmp = loc(bound.xy)
        if loctmp[2] - loctmp[0] != 1200 or loctmp[3] - loctmp[1] != 1200 or len(bound.xy) != 5:
            errnum += 1
            print ('File Name: img/%s/%d_%d_%d.png'%(dataclass, name, ptype, nums[ptype]))
            print ('Err Bound: ', bound.xy)
            print (loctmp, loctmp[2] - loctmp[0], loctmp[3] - loctmp[1])
            # continue
        nums[ptype] += 1
    print('Bound Errs: %d'%errnum)
    print('Hotspots: %d, Non-hotspots: %d'%(nums[1], nums[0]))

if __name__ == "__main__":
    dataclass = sys.argv[1]
    dataname = int(sys.argv[2])
    cutpic(dataclass, dataname)
    # check_bound_err(dataclass, dataname)
    