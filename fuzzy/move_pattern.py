from gdsii.library import Library
from gdsii.elements import *
import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys
import random
import os

def read_patterns(file_name):
    patterns = []
    with open(file_name) as f:
        for line in f:
            intline = list(map(lambda x:int(x), line.split()))
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
    # move_obj = random.randint(0, len(patterns)-1)
    # print(move_obj)
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
    # print(move_dist)
    move_dist = int(move_dist)
    if min_direction == 'h':
        patterns[move_obj] += np.array([0, move_dist])
    elif min_direction == 'v':
        patterns[move_obj] += np.array([move_dist, 0])

# coord_name = 'coordinate/train/1_1_0.txt'
def generate_one(coord_name, work_path, frac=0.25, ):
    hotspot = coord_name.split('/')[-1].split('_')[1]=='1'
    patterns = read_patterns(os.path.join(work_path, coord_name))
    move_poly(patterns, frac, hotspot)
    img = np.zeros([4800,4800], dtype=np.uint8)
    for pattern in patterns:
        cv2.fillPoly(img, [pattern[:-1]], 255)
    img_out = np.zeros([2048,2048], dtype=np.uint8)
    img_out[512:3*512, 512:3*512] = img[1800:2824, 1800:2824]
    cv2.imwrite(os.path.join(work_path, coord_name.replace('txt', 'png')) ,img_out)

if __name__ == "__main__":
    txt_path = sys.argv[1]
    for coord_file in os.listdir(txt_path):
        if coord_file.split('.')[-1] == 'txt':
            # print(coord_file)
            generate_one(coord_file, txt_path, 0.2)
