
import random
import math as m
import numpy as np
import sys
import cv2


def stdv(X):
    if not len(X):
        return 0
    sum_num = 0
    for t in X:
        sum_num = sum_num + t           
    mean = sum_num / len(X)
    tot = 0.0
    for x in X:
        tot = tot + (x - mean)**2
    return (tot/len(X))**0.5

# Function to find equation of plane. 
def equation_plane(p1, p2, p3):  
    
    x1 = p1[0]; x2 = p2[0]; x3 = p3[0]
    y1 = p1[1]; y2 = p2[1]; y3 = p3[1]
    z1 = p1[2]; z2 = p2[2]; z3 = p3[2]

    a1 = x2 - x1 
    b1 = y2 - y1 
    c1 = z2 - z1 
    a2 = x3 - x1 
    b2 = y3 - y1 
    c2 = z3 - z1 
    a = b1 * c2 - b2 * c1 
    b = a2 * c1 - a1 * c2 
    c = a1 * b2 - b1 * a2 
    d = (- a * x1 - b * y1 - c * z1) 

    return [a,b,c,d]

def plane_to_xy_angle(plane):
    denom = m.sqrt((plane[0]*plane[0]) + (plane[1]*plane[1]) + (plane[2]*plane[2]))
    if denom:
        cos = plane[2]/(denom)
        theta = m.degrees(m.acos(cos))
        return theta
    return 361


def distance_plane_to_point(plane, point):

    num = abs((plane[0] * point[0]) + (plane[1] * point[1]) + (plane[2] * point[2]) + plane[3])
    den = m.sqrt((plane[0]*plane[0]) + (plane[1]*plane[1]) + (plane[2]*plane[2]))
    if den < 0.00000001:
        return 999999

    return float(num/den)


def do_ransac(pc_list):
    
    best_support = 0
    best_index = [np.int(x) for x in range(0)]
    best_plane = [0,0,0,0]
    bestStd = 99999999999999
    i = 0
    e = 1 - (60/100)
    alpha = 0.90
    N =  40 # round(m.log(1-alpha)/m.log(1-m.pow((1-e),3)))

    if (len(pc_list)) < 10:
        return None
    while i < N:
        p1 = random.randint(0, len(pc_list)-1)
        p2 = random.randint(0, len(pc_list)-1)
        while p1 == p2:
            p2 = random.randint(0, len(pc_list)-1)
        p3 = random.randint(0, len(pc_list)-1) 
        while p1 == p3 or p2 == p3:
            p3 = random.randint(0, len(pc_list)-1)

        pl = equation_plane(pc_list[p1], pc_list[p2], pc_list[p3])
        if not (detect_land(pl)):
            continue

        index = []
        s = []
        t = 0.1
        for a in range(len(pc_list)):
            distance_to_plane = distance_plane_to_point(pl, pc_list[a])
            if distance_to_plane < t:
                s.append(distance_to_plane)
                index.append(a)
        if not s:
            continue
        st = stdv(s)

        
        if (len(s) > best_support) or (len(s) == best_support and st < bestStd):
            best_support = len(s)
            best_plane = pl
            bestStd = st
            best_index = index

        i = i +1

    # land_detected = detect_land(best_plane)
    
    # if (land_detected):
    if best_support > len(pc_list) * 0.25:
        return best_index
    else:
        return None


def detect_land(plane):
    angle = plane_to_xy_angle(plane)
    if 160 < angle and angle < 200:
        return True
    if 0 < angle and angle < 30:
        return True

    return False
