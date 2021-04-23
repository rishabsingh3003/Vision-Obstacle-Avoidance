from numba import njit
import random
import math as m
import numpy as np
import time

last_run_time = 0 # last time the algorithm was run
last_update_time = 0 # last time algorithm returned valid equation
last_ground_eqn = np.array([0.0, 0.0, 0.0, 0.0]) # last valid equation
last_valid = False

run_time = 0.3 #run every this many seconds, we don't expect the ground equation to change too quickly

# run the ransac algorithm
def run(pc, land_detect_area, step_x_size = 0, step_y_size = 0):
    global last_run_time, last_update_time, last_ground_eqn, last_valid

    dt = time.time() - last_run_time
    if (dt > run_time):
        last_run_time = time.time()
        ground_eqn, valid_eqn = do_ransac(pc, land_detect_area, step_x_size, step_y_size)
        dt_update = time.time() - last_update_time
        if (not valid_eqn and dt_update < 2):
            return last_ground_eqn, valid_eqn
        else:
            last_ground_eqn = ground_eqn
            last_valid = valid_eqn
            last_update_time = time.time()
            return last_ground_eqn, valid_eqn
    return last_ground_eqn, last_valid


# figure out three random points from the land_detection_area of the passed frame
@njit
def pc_to_three_points(pc, land_detect_area):

    width = pc.shape[0]
    height = pc.shape[1]
    random_points = np.zeros((3, 3))
    index = 0
    counter = 0
    while (index < 3):
        random_index1 = random.randint(int(width*land_detect_area), width-1)
        random_index2 = random.randint(0, height-1)
        counter = counter + 1
        point = pc[random_index1, random_index2]
        if (point[0]):
            for last_pt in random_points:
                if ((last_pt == point).all()):
                    continue
            random_points[index] = point
            index = index + 1
        if (counter >= 1000):
            # cannot find 3 unique points in over 1000 iterations, lets exit
            return random_points, False

    return random_points, True


# RANSAC algorithm, returns the equation of the plane and if its valid or not
@njit
def do_ransac(pc, land_detect_area, step_x_size = 0, step_y_size = 0):

    width = pc.shape[0]
    height = pc.shape[1]

    best_support = 0
    best_plane = np.array([0.0, 0.0, 0.0, 0.0])
    bestStd = 99999999999999
    i = 0
    e = 1 - (60/100)
    alpha = 0.90
    N =  40 # round(m.log(1-alpha)/m.log(1-m.pow((1-e),3)))

    while i < N:
        random_points, valid_points = pc_to_three_points(pc, land_detect_area)
        if (valid_points == False):
            # can't find random points, exit immediately
            return best_plane, False
        pl = equation_plane(random_points[0], random_points[1], random_points[2])
        if not (detect_land(pl)):
            continue
        s = []
        t = 0.1
        for u in range(int(width*land_detect_area), width, step_x_size):
            for v in range(0, height, step_y_size):
                point = pc[u,v]
                if (point[0] == 0):
                    continue
                distance_to_plane = distance_plane_to_point(pl, point)
                if distance_to_plane < t:
                    s.append(distance_to_plane)
            if not s:
                continue
            st = stdv(s)
        if (len(s) > best_support) or (len(s) == best_support and st < bestStd):
            best_support = len(s)
            best_plane = pl
            bestStd = st

        i = i +1

    if best_support > 10 and bestStd < 0.026:
        return best_plane, True
    else:
        return best_plane, False



#----------------------- MATH FUNCTIONS OPTIMIZED WITH NUMBA-----------------------#



# Function to find equation of plane.
@njit
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

    return np.array([a,b,c,d])

# distance of plane to given point
@njit
def distance_plane_to_point(plane, point):

    num = abs((plane[0] * point[0]) + (plane[1] * point[1]) + (plane[2] * point[2]) + plane[3])
    den = m.sqrt((plane[0]*plane[0]) + (plane[1]*plane[1]) + (plane[2]*plane[2]))
    if den < 0.00000001:
        return 999999

    return float(num/den)

# standard deviation of list
@njit
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

# we expect ground to be less than 20-30 degree's with respect to the ground axis
@njit
def detect_land(plane):
    angle = plane_to_xy_angle(plane)
    if 150 < angle and angle < 180:
        return True
    if 0 < angle and angle < 20:
        return True

    return False

# returns the angle plane makes with ground axis
@njit
def plane_to_xy_angle(plane):
    denom = m.sqrt((plane[0]*plane[0]) + (plane[1]*plane[1]) + (plane[2]*plane[2]))
    if denom:
        cos = plane[2]/(denom)
        theta = m.degrees(m.acos(cos))
        return theta
    return 361