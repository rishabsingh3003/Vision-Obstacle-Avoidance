# Install required packages: 
# 
#   pip3 install pymavlink
#   pip3 install apscheduler
#   pip3 install opencv-python
#   pip3 install airsim
#   pip3 install numpy
#   sudo apt-get install python-PIL
import os
os.environ["MAVLINK20"] = "1"
import math
import sys
import time
import airsim
from cv2 import cv2
import numpy as np
from PIL import Image
import threading
from time import sleep
from apscheduler.schedulers.background import BackgroundScheduler
from pymavlink import mavutil
import argparse

sys.path.append("/usr/local/lib/")



######################################################
##  Parsing user' inputs                            ##
######################################################

parser = argparse.ArgumentParser(description='Reboots vehicle')
parser.add_argument('--connect',
                    help="Vehicle connection target string. If not specified, a default string will be used.")
parser.add_argument('--baudrate', type=float,
                    help="Vehicle connection baudrate. If not specified, a default value will be used.")
parser.add_argument('--obstacle_distance_msg_hz', type=float,
                    help="Update frequency for OBSTACLE_DISTANCE message. If not specified, a default value will be used.")

args = parser.parse_args()

# Default configurations for connection to the FCU
if not args.connect:
    connection_string = 'localhost:14551'
else:
    connection_string = args.connect

if not args.baudrate:
    connection_baudrate = 921600
else:
    connection_baudrate = args.baudrate

if not args.obstacle_distance_msg_hz:
    obstacle_distance_msg_hz = 15
else:
    obstacle_distance_msg_hz = args.obstacle_distance_msg_hz

# AirSim API
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)

DEPTH_RANGE_M = [0.3, 30] # depth range, to be changed as per requirements
MAX_DEPTH = 9999 # arbitrary large number 

#numpy array to share obstacle coordinates between main thread and mavlink thread
mavlink_obstacle_coordinates = np.ones((9,3), dtype = np.float) * (MAX_DEPTH)

# get time in correct format
start_time =  int(round(time.time() * 1000))
current_milli_time = lambda: int(round(time.time() * 1000) - start_time)

# get depth from airsim backend
def get_depth(client):
    requests = []
    requests.append(airsim.ImageRequest(
                'front_center', airsim.ImageType.DepthPlanner, pixels_as_float=True, compress=False))

    responses = client.simGetImages(requests)
    depth = airsim.list_to_2d_float_array(
                    responses[0].image_data_float, responses[0].width, responses[0].height)
    depth = np.expand_dims(depth, axis=2)
    depth = depth.squeeze()
    return depth, responses[0].width, responses[0].height

# this method converts, (x,y) from depth matrix to NEU 3-D vector in body frame
def convert_depth_3D_vec(x_depth, y_depth, depth, fov):
    # https://stackoverflow.com/questions/62046666/find-3d-coordinate-with-respect-to-the-camera-using-2d-image-coordinates
    h, w = depth.shape
    center_x = w // 2
    center_y = h // 2
    focal_len = w / (2 * np.tan(fov / 2))
    x = depth[y_depth, x_depth]
    y = (x_depth - center_x) * x / focal_len
    z = -1 * (y_depth - center_y) * x / focal_len
    return x,y,z

# divide the depth data into a 3x3 grid. Pick out the smallest distance in each grid
# store the x,y of the depth matrix, the 9 depths, and convert them into body-frame x,y,z
def distances_from_depth_image(depth_mat, min_depth_m, max_depth_m, depth, depth_coordinates, obstacle_coordinates, valid_depth):
    # Parameters for depth image
    depth_img_width  = depth_mat.shape[1]
    depth_img_height = depth_mat.shape[0]
    
    # Parameters for obstacle distance message
    step_x = depth_img_width / 20
    step_y = depth_img_height/ 20

    
    sampling_width = int(1/3 * depth_img_width)
    sampling_height = int(1/3* depth_img_height)
    # divide the frame into 3x3 grid to find the minimum depth value in "9 boxes"
    for i in range(9):
        if i%3 == 0 and i != 0: 
            sampling_width = int(1/3* depth_img_width)
            sampling_height = sampling_height + int(1/3 * depth_img_height)

        x,y = 0,0
        x = sampling_width - int(1/3 * depth_img_width)

        while x < sampling_width:
            x = x + step_x
            y = sampling_height - int(1/3* depth_img_height)
            while y < sampling_height:
                y = y + step_y
                # make sure coordinates stay within matrix limits
                x_pixel = 0 if x < 0 else depth_img_width-1 if x > depth_img_width -1 else int(x)
                y_pixel = 0 if y < 0 else depth_img_height-1 if y > depth_img_height -1 else int(y)
                
                #convert depth to body-frame x,y,z 
                x_obj,y_obj,z_obj = convert_depth_3D_vec(x_pixel, y_pixel, depth_mat, math.radians(90))
                
                # actual euclidean distance to obstacle
                point_depth = (x_obj*x_obj + y_obj*y_obj + z_obj*z_obj)**0.5
                
                # if within valid range, mark this depth as valid and store all the info
                if point_depth <= depth[i] and point_depth > min_depth_m and point_depth < max_depth_m:
                    depth[i] = point_depth
                    depth_coordinates[i] = [x_pixel,y_pixel]
                    obstacle_coordinates[i] = [x_obj, y_obj, z_obj]
                    valid_depth[i] = True
        
        sampling_width = sampling_width + int(1/3* depth_img_width)
        
# display depth image from AirSim. The data received from AirSim needs to be proceeded for a good view 
# also divides the view into 3x3 grid and prints smallest depth value in each grid
# puts a circle on the smallest found depth value
def getScreenDepthVis(client, depth_coordinates, depth_list):
    #get image from airsim
    responses = client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.DepthPlanner, True, False)])
    # condition the data
    img1d = np.array(responses[0].image_data_float, dtype=np.float)
    img1d = 255/np.maximum(np.ones(img1d.size), img1d)
    img2d = np.reshape(img1d, (responses[0].height, responses[0].width))
    image = np.invert(np.array(Image.fromarray(img2d.astype(np.uint8), mode='L')))
    
    factor = 10
    maxIntensity = 255.0 # depends on dtype of image data
    
    # Decrease intensity such that dark pixels become much darker, bright pixels become slightly dark 
    newImage1 = (maxIntensity)*(image/maxIntensity)**factor
    newImage1 = newImage1.astype(np.uint8)
                                          
    color_img = cv2.applyColorMap(newImage1, cv2.COLORMAP_JET)
    # divide view into 3x3 matrix
    pxstep = int(newImage1.shape[1]/3)
    pystep = int(newImage1.shape[0]/3)
    gx = pxstep
    gy = pystep
    while gx < newImage1.shape[1]:
        cv2.line(color_img, (gx, 0), (gx, newImage1.shape[0]), color=(0, 0, 0), thickness=1)
        gx += pxstep
    while gy < newImage1.shape[0]:
        cv2.line(color_img, (0, gy), (newImage1.shape[1], gy), color=(0, 0, 0),thickness=1)
        gy += pystep
    
    # print circle, and depth values on the screen
    for i in range(len(depth_list)):
        if depth_list[i] <= DEPTH_RANGE_M[1]:
            color_img = cv2.circle(color_img, (int(depth_coordinates[i][0]),int(depth_coordinates[i][1])), 5, (0, 0, 0), 5)
            color_img = cv2.putText(color_img, str(round(depth_list[i],2)), (int(pxstep*(1/4 + i%3)),int(pystep*(1/3 + math.floor(i/3)))), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)
    
    cv2.imshow("Depth Vis", color_img)
    cv2.waitKey(1)


def mavlink_loop(conn, callbacks):
    '''a main routine for a thread; reads data from a mavlink connection,
    calling callbacks based on message type received.
    '''
    interesting_messages = list(callbacks.keys())
    while True:
        # send a heartbeat msg
        conn.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                                mavutil.mavlink.MAV_AUTOPILOT_GENERIC,
                                0,
                                0,
                                0)
        m = conn.recv_match(type=interesting_messages, timeout=1, blocking=True)
        if m is None:
            continue
        callbacks[m.get_type()](m)

# send mavlink message to SITL
def send_obstacle_distance_3D_message():
    global conn
    time = current_milli_time()
    # print(mavlink_obstacle_coordinates)
    for i in range(9):
        conn.mav.obstacle_distance_3d_send(
            time,    # us Timestamp (UNIX time or time since system boot)
            0,       # not implemented in ArduPilot            
            12,       # not implemented in ArduPilot           
            65535,   # unknown ID of the object. We are not really detecting the type of obstacle           
            float(mavlink_obstacle_coordinates[i][0]),	   # X in NEU body frame 
            float(mavlink_obstacle_coordinates[i][1]),     # Y in NEU body frame  
            float(-mavlink_obstacle_coordinates[i][2]),	   # Z in NEU body frame  
            float(DEPTH_RANGE_M[0]), # min range of sensor
            float(DEPTH_RANGE_M[1])  # max range of sensor
        )

conn = mavutil.mavlink_connection(
    device = str(connection_string),
    autoreconnect = True,
    source_system = 1,
    source_component = 93,
    baud=connection_baudrate,
    force_connected=True,
)

mavlink_callbacks = {
}
mavlink_thread = threading.Thread(target=mavlink_loop, args=(conn, mavlink_callbacks))
mavlink_thread.start()

# Send MAVlink messages in the background at pre-determined frequencies
sched = BackgroundScheduler()
sched.add_job(send_obstacle_distance_3D_message, 'interval', seconds = 1/obstacle_distance_msg_hz)
sched.start()

# main loop
while True:
    #depth image from airsim
    depth_mat,width,height = get_depth(client)
    # Will be populated with smallest depth in each grid 
    depth_list = np.ones((9,), dtype=np.float) * (DEPTH_RANGE_M[1] + 1)
    # Valid depth in each grid will be marked True
    valid_depth = np.ones((9,), dtype=np.bool) * False
    # Matrix Coordinated of the smallest depth in each grid
    depth_coordinates = np.ones((9,2), dtype=np.uint16) * (MAX_DEPTH) 
    # Body frame NEU XYZ coordinates of obstacles to be sent to vehicle
    obstacle_coordinates = np.ones((9,3),dtype=np.float) * (MAX_DEPTH)
    
    #get the obstacles
    distances_from_depth_image(depth_mat, DEPTH_RANGE_M[0], DEPTH_RANGE_M[1], depth_list, depth_coordinates, obstacle_coordinates, valid_depth)
    # if valid, populate mavlink array
    for i in range(9):
        if valid_depth[i]:
            mavlink_obstacle_coordinates[i] = obstacle_coordinates[i]
        else:
            mavlink_obstacle_coordinates[i] = MAX_DEPTH
    
    # visualize the data
    getScreenDepthVis(client, depth_coordinates, depth_list)
    
