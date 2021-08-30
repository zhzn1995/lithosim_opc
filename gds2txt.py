from gdsii.library import Library
from gdsii.elements import *
import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys
import os

def loc(x):
    i=np.array(x)
    return np.array([i[:,0].min(), i[:,1].min(), i[:,0].max(), i[:,1].max()])
def contain(loc1, loc2):
    return loc1[0]<=loc2[0] and loc1[1]<=loc2[1] and loc1[2]>=loc2[2] and loc1[3]>=loc2[3]

def seg_type(seg):
    if seg[0,0] == seg[1,0]:
        return 'v'
    elif seg[0,1] == seg[1,1]:
        return 'h'
    else:
        raise Exception('illegal segment!')

def if_seg_cut(seg1, seg2):
    type1 = seg_type(seg1)
    type2 = seg_type(seg2)
    if type1 == type2:
        return False
    elif type1 == 'v':
        x = seg1[0,0]
        x1 = seg2[0,0]
        x2 = seg2[1,0]

        y = seg2[0,1]
        y1 = seg1[0,1]
        y2 = seg1[1,1]
    else:
        x = seg2[0,0]
        x1 = seg1[0,0]
        x2 = seg1[1,0]

        y = seg1[0,1]
        y1 = seg2[0,1]
        y2 = seg2[1,1]
    if (x-x1)*(x-x2)<0 and (y-y1)*(y-y2)<0:
        return True
    else:
        return False

def if_seg_cut_polygon(seg, poly):
    len_poly = len(poly)
    for i in range(len_poly-1):
        poly_seg = poly[i:i+2]
        if if_seg_cut(seg, poly_seg):
            return True
    return False


def cutpic(dataclass, name):
    if dataclass == "train":
        filename = os.path.join(work_path, 'raw/MX_Benchmark%d_clip.gds'%name)
    elif dataclass == "val":
        filename = os.path.join(work_path, 'raw/array_benchmark%d.gds'%name)
    
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
        if bound.layer%10 in range(1,3):
            ptype=1
        elif bound.layer%10==3:
            ptype=0
        else:
            print('error!')
        filename = os.path.join(work_path, 'coordinate/%s/%d_%d_%d.txt'%(dataclass, name, ptype, nums[ptype]))
        f = open(filename, 'w')

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
            if_cut = 0
            pat = np.array(content, dtype=np.int64)
            poly = pat-base
            edges = [[[1800,1800],[2824,1800]],[[2824,1800],[2824,2824]],[[2824,2824],[1800,2824]],[[1800,2824],[1800,1800]]]
            if (poly.min(axis=0)>np.array([1800,1800])).mean()==1 and (poly.max(axis=0)<np.array([2824,2824])).mean()==1:
                for i in range(len(poly)):
                    f.write('%d %d '%(poly[i,0], poly[i,1]))
                f.write('\n')
            else:
                for edge in edges:
                    if if_seg_cut_polygon(np.array(edge), poly):
                        if_cut = 1
                if if_cut:
                    for i in range(len(poly)):
                        f.write('%d %d '%(poly[i,0], poly[i,1]))
                    f.write('\n')
                # cv2.fillPoly(img, [poly[:-1]], 255)
        f.close()
            # continue
        # img_out = np.zeros([2048,2048], dtype=np.uint8)
        # img_out[512:3*512, 512:3*512] = img[1800:2824, 1800:2824]
        # cv2.imwrite('img_test/%s/%d_%d_%d.png'%(dataclass, name, ptype, nums[ptype]) ,img)
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
    work_path = sys.argv[3]
    cutpic(dataclass, dataname)
    # check_bound_err(dataclass, dataname)
    