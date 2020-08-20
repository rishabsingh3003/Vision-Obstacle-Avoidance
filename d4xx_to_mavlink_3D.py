import sys
sys.path.append("/usr/local/lib/")

# Set MAVLink protocol to 2.
import os
os.environ["MAVLINK20"] = "1"

# Import the libraries
import pyrealsense2 as rs
import numpy as np
import math as m
import time
import argparse
import threading
import json
from apscheduler.schedulers.background import BackgroundScheduler
from pymavlink import mavutil
import cv2
import plotly.graph_objs as go

######################################################
##  Depth parameters - reconfigurable               ##
######################################################

# Sensor-specific parameter, for D435: https://www.intelrealsense.com/depth-camera-d435/
STREAM_TYPE = [rs.stream.depth, rs.stream.color]  # rs2_stream is a types of data provided by RealSense device
FORMAT      = [rs.format.z16, rs.format.bgr8]     # rs2_format is identifies how binary data is encoded within a frame
DEPTH_WIDTH = 640               # Defines the number of columns for each frame or zero for auto resolve
DEPTH_HEIGHT= 480               # Defines the number of lines for each frame or zero for auto resolve
FPS         = 30                # Defines the rate of frames per second
DEPTH_RANGE_M = [0.1, 8.0]        # Replace with your sensor's specifics, in meter

USE_PRESET_FILE = True
PRESET_FILE  = "../cfg/d4xx-default.json"

# List of filters to be applied, in this order.
# https://github.com/IntelRealSense/librealsense/blob/master/doc/post-processing-filters.md

filters = [
    [True, "Decimation Filter",     rs.decimation_filter()],
    [True, "Threshold Filter",      rs.threshold_filter()],
    [True, "Depth to Disparity",    rs.disparity_transform(True)],
    [True, "Spatial Filter",        rs.spatial_filter()],
    [True, "Temporal Filter",       rs.temporal_filter()],
    [False, "Hole Filling Filter",   rs.hole_filling_filter()],
    [True, "Disparity to Depth",    rs.disparity_transform(False)]
]

#
# The filters can be tuned with opencv_depth_filtering.py script, and save the default values to here
# Individual filters have different options so one have to apply the values accordingly
#

# decimation_magnitude = 8
# filters[0][2].set_option(rs.option.filter_magnitude, decimation_magnitude)

threshold_min_m = DEPTH_RANGE_M[0]
threshold_max_m = DEPTH_RANGE_M[1]
if filters[1][0] is True:
    filters[1][2].set_option(rs.option.min_distance, threshold_min_m)
    filters[1][2].set_option(rs.option.max_distance, threshold_max_m)

######################################################
##  ArduPilot-related parameters - reconfigurable   ##
######################################################

# Default configurations for connection to the FCU
connection_string_default = '127.0.0.1:14551'
connection_baudrate_default = 921600

# Enable/disable each message/function individually
obstacle_distance_msg_hz_default = 15.0

# lock for thread synchronization
lock = threading.Lock()

debug_enable_default = 0

######################################################
##  Global variables                                ##
######################################################

# FCU connection variables
vehicle = None
is_vehicle_connected = False
vehicle_pitch_rad = None

# Camera-related variables
pipe = None
depth_scale = 0
depth_intrinsics = None

current_time_us = 0

# Obstacle distances in front of the sensor, starting from the left in increment degrees to the right
# See here: https://mavlink.io/en/messages/common.html#OBSTACLE_DISTANCE
min_depth = DEPTH_RANGE_M[0]  
max_depth = DEPTH_RANGE_M[1]  
distances_array_length = 72
x_obstacles,y_obstacles,z_obstacles, obstacle_ID = [0]*9,[0]*9,[0]*9, [0]*9

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
parser.add_argument('--debug_enable',type=float,
                    help="Enable debugging information")

args = parser.parse_args()

connection_string = args.connect
connection_baudrate = args.baudrate
obstacle_distance_msg_hz = args.obstacle_distance_msg_hz
debug_enable = args.debug_enable

# Using default values if no specified inputs
if not connection_string:
    connection_string = connection_string_default
    print("INFO: Using default connection_string", connection_string)
else:
    print("INFO: Using connection_string", connection_string)

if not connection_baudrate:
    connection_baudrate = connection_baudrate_default
    print("INFO: Using default connection_baudrate", connection_baudrate)
else:
    print("INFO: Using connection_baudrate", connection_baudrate)
    
if not obstacle_distance_msg_hz:
    obstacle_distance_msg_hz = obstacle_distance_msg_hz_default
    print("INFO: Using default obstacle_distance_msg_hz", obstacle_distance_msg_hz)
else:
    print("INFO: Using obstacle_distance_msg_hz", obstacle_distance_msg_hz)

# The list of filters to be applied on the depth image
for i in range(len(filters)):
    if filters[i][0] is True:
        print("INFO: Applying: ", filters[i][1])
    else:
        print("INFO: NOT applying: ", filters[i][1])

if not debug_enable:
    debug_enable = debug_enable_default

if debug_enable == 1:
    print("INFO: Debugging option enabled")
else:
    print("INFO: Debugging option DISABLED")


######################################################
##  Functions - D4xx cameras                        ##
######################################################

DS5_product_ids = ["0AD1", "0AD2", "0AD3", "0AD4", "0AD5", "0AF6", "0AFE", "0AFF", "0B00", "0B01", "0B03", "0B07","0B3A"]

def find_device_that_supports_advanced_mode() :
    ctx = rs.context()
    ds5_dev = rs.device()
    devices = ctx.query_devices()
    for dev in devices:
        if dev.supports(rs.camera_info.product_id) and str(dev.get_info(rs.camera_info.product_id)) in DS5_product_ids:
            if dev.supports(rs.camera_info.name):
                print("INFO: Found device that supports advanced mode:", dev.get_info(rs.camera_info.name))
            return dev
    raise Exception("No device that supports advanced mode was found")

# Loop until we successfully enable advanced mode
def realsense_enable_advanced_mode(advnc_mode):
    while not advnc_mode.is_enabled():
        print("INFO: Trying to enable advanced mode...")
        advnc_mode.toggle_advanced_mode(True)
        # At this point the device will disconnect and re-connect.
        print("INFO: Sleeping for 5 seconds...")
        time.sleep(5)
        # The 'dev' object will become invalid and we need to initialize it again
        dev = find_device_that_supports_advanced_mode()
        advnc_mode = rs.rs400_advanced_mode(dev)
        print("INFO: Advanced mode is", "enabled" if advnc_mode.is_enabled() else "disabled")

# Load the settings stored in the JSON file
def realsense_load_settings_file(advnc_mode, setting_file):
    # Sanity checks
    if os.path.isfile(setting_file):
        print("INFO: Setting file found", setting_file)
    else:
        print("INFO: Cannot find setting file ", setting_file)
        exit()

    if advnc_mode.is_enabled():
        print("INFO: Advanced mode is enabled")
    else:
        print("INFO: Device does not support advanced mode")
        exit()
    
    # Input for load_json() is the content of the json file, not the file path
    with open(setting_file, 'r') as file:
        json_text = file.read().strip()

    advnc_mode.load_json(json_text)

# Establish connection to the Realsense camera
def realsense_connect():
    global pipe, depth_scale
    # Declare RealSense pipe, encapsulating the actual device and sensors
    pipe = rs.pipeline()

    # Configure depth and color streams
    config = rs.config()
    config.enable_stream(STREAM_TYPE[0], DEPTH_WIDTH, DEPTH_HEIGHT, FORMAT[0], FPS)
    
    config.enable_stream(STREAM_TYPE[1], DEPTH_WIDTH, DEPTH_HEIGHT, FORMAT[1], FPS)
    
    # Start streaming with requested config
    profile = pipe.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("INFO: Depth scale is: ", depth_scale)

def realsense_configure_setting(setting_file):
    device = find_device_that_supports_advanced_mode()
    advnc_mode = rs.rs400_advanced_mode(device)
    realsense_enable_advanced_mode(advnc_mode)
    realsense_load_settings_file(advnc_mode, setting_file)

# Setting parameters for the OBSTACLE_DISTANCE message based on actual camera's intrinsics and user-defined params
def set_obstacle_distance_params():
    global depth_scale, depth_intrinsics
    
    # Obtain the intrinsics from the camera itself
    profiles = pipe.get_active_profile()
    depth_intrinsics = profiles.get_stream(STREAM_TYPE[0]).as_video_stream_profile().intrinsics
    print("INFO: Depth camera intrinsics: ", depth_intrinsics)

def send_obstacle_message():
    connection.mav.viso_obstacle_distance_send(0, 0, obstacle_ID, x_obstacles, y_obstacles, z_obstacles, 0.5, 10)
    
def convert_depth_to_phys_coord_using_realsense(x, y, depth, cameraInfo):
  result = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], depth) 
  
  center_pixel =  [depth_intrinsics.ppy/2,depth_intrinsics.ppx/2]
  result_center = rs.rs2_deproject_pixel_to_point(depth_intrinsics,center_pixel, depth)
  
  return result[2], (result[1] - result_center[1]), -(result[0]- result_center[0])


class quadrant_data:

    distance = 65535
    def __init__(self, width_start, width_end, height_start, height_end):
        self.width_start = width_start
        self.width_end = width_end
        self.height_start = height_start
        self.height_end = height_end

    def point_in_section(self,depth_mat, x, y):
        depth_distance = depth_mat[int(x), int(y)]
        depth_distance = depth_distance * depth_scale 
        if depth_distance < min_depth:
            return 65535
        if depth_distance > max_depth:
            return 65535
        return depth_distance
    
    def minimum_distance_in_section(self, step_horizontal, step_vertical, depth_mat):
        distance = 65535
        point_distance_packet = [65535,0,0]
        for i in range(int(self.width_start), int(self.width_end), int(step_horizontal)):
            for j in range(int(self.height_start), int(self.height_end), int(step_vertical)):
                point_distance = self.point_in_section(depth_mat, j, i)
                if point_distance == 65535:
                    continue
                if point_distance < distance:
                    distance = point_distance
                    point_distance_packet = [i,j,distance]
        return point_distance_packet

class obstacles_in_frame:

    def __init__(self,no_of_rows_or_column, frame_width, frame_height):

        fh, fh1by3, fh2by3 = int(frame_height), int(frame_height/3), int(frame_height*2/3)
        fw, fw1by3, fw2by3 = int(frame_width), int(frame_width/3), int(frame_width*2/3)

        b1x1 = quadrant_data(1,fw1by3,1,fh1by3)           #0
        b1x2 = quadrant_data(fw1by3, fw2by3, 1, fh1by3)   #1
        b1x3 = quadrant_data(fw2by3, fw, 1, fh1by3)       #2
        b2x1 = quadrant_data(1,fw1by3,fh1by3,fh2by3)      #3
        b2x2 = quadrant_data(fw1by3,fw2by3,fh1by3,fh2by3) #4
        b2x3 = quadrant_data(fw2by3,fw,fh1by3,fh2by3)     #5
        b3x1 = quadrant_data(1,fw1by3,fh2by3,fh)          #6
        b3x2 = quadrant_data(fw1by3,fw2by3,fh2by3,fh)     #7
        b3x3 = quadrant_data(fw2by3,fw,fh2by3,fh)         #8
        self.quadrant_list = [b1x1, b1x2, b1x3, b2x1, b2x2, b2x3, b3x1, b3x2, b3x3]

    def update_quadrants(self, depth_mat, step_horizontal, step_vertical):
        quadrant_distance_list = []
        for box in self.quadrant_list:
            quadrant_distance_list.append(box.minimum_distance_in_section(int(step_horizontal), int(step_vertical) ,depth_mat))
        return quadrant_distance_list


def distance_from_depth(obstacle, depth_mat,min_depth_m, max_depth_m):
    # Parameters for depth image
    depth_img_width  = depth_mat.shape[1]
    depth_img_height = depth_mat.shape[0]
    step_horizontal = depth_img_width / distances_array_length
    step_vertical = depth_img_height/ distances_array_length
    obstacles = obstacles_in_frame(3, depth_img_width, depth_img_height)
    obstacles_list = obstacles.update_quadrants(depth_mat, step_horizontal, step_vertical)
    return obstacles_list

# plot obstacle vectors
def vector_plot(tvects,is_vect=True,orig=[0,0,0]):
    """Plot vectors using plotly"""

    if is_vect:
        if not hasattr(orig[0],"__iter__"):
            coords = [[orig,np.sum([orig,v],axis=0)] for v in tvects]
        else:
            coords = [[o,np.sum([o,v],axis=0)] for o,v in zip(orig,tvects)]
    else:
        coords = tvects

    data = []
    for i,c in enumerate(coords):
        X1, Y1, Z1 = zip(c[0])
        X2, Y2, Z2 = zip(c[1])
        vector = go.Scatter3d(x = [X1[0],X2[0]],
                              y = [Y1[0],Y2[0]],
                              z = [Z1[0],Z2[0]],
                              marker = dict(size = [0,5],
                                            color = ['blue'],
                                            line=dict(width=5,
                                                      color='DarkSlateGrey')),
                              name = 'Vector'+str(i+1))
        data.append(vector)

    layout = go.Layout(
             margin = dict(l = 4,
                           r = 4,
                           b = 4,
                           t = 4)
                  )
    fig = go.Figure(data=data,layout=layout)
    fig.show()

######################################################
##  Main code starts here                           ##
######################################################

print('Connecting to camera...')
if USE_PRESET_FILE:
    realsense_configure_setting(PRESET_FILE)
realsense_connect()
print('Camera connected.')

set_obstacle_distance_params()
connection = mavutil.mavlink_connection('udpin:localhost:14551')
print('Waiting for heartbeat from vehicle...')
connection.wait_heartbeat()
print('Vehicle Connected')
sched = BackgroundScheduler()
sched.add_job(send_obstacle_message, 'interval', seconds = 1/obstacle_distance_msg_hz)
sched.start()

try:
    while True:
        # This call waits until a new coherent set of frames is available on a device
            # Calls to get_frame_data(...) and get_frame_timestamp(...) on a device will return stable values until wait_for_frames(...) is called
        frames = pipe.wait_for_frames() 
        
        align = rs.align(rs.stream.color)
        frames = align.process(frames)

        depth_frame = frames.get_depth_frame()

        if not depth_frame:
            continue
        
        # Apply the filters
        filtered_frame = depth_frame
        for i in range(len(filters)):
            if filters[i][0] is True:
                filtered_frame = filters[i][2].process(filtered_frame)

        # Extract depth in matrix form
        depth_data = filtered_frame.as_frame().get_data()
        depth_mat = np.asanyarray(depth_data)

        obstacle_list = distance_from_depth(1, depth_mat,0,10)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_mat, alpha=0.03), cv2.COLORMAP_JET)

        # divide view into 3x3 matrix
        pxstep = int(depth_mat.shape[1]/3)
        pystep = int(depth_mat.shape[0]/3)
        gx = pxstep
        gy = pystep
        while gx < depth_mat.shape[1]:
            cv2.line(depth_colormap, (gx, 0), (gx, depth_mat.shape[0]), color=(0, 0, 0), thickness=1)
            gx += pxstep
        while gy < depth_mat.shape[0]:
            cv2.line(depth_colormap, (0, gy), (depth_mat.shape[1], gy), color=(0, 0, 0),thickness=1)
            gy += pystep

        obstacle_vector = []
        for i in range(len(obstacle_list)): 
            if obstacle_list[i][2] == 65535:
                x,y,z = 65535,0,0
            else:
                if obstacle_list[i][2] < 1.5:
                    # put circle on detected obstacle
                    cv2.circle(depth_colormap, (obstacle_list[i][0],obstacle_list[i][1]), 15, color = (255,0, 255), thickness = 3)
                # put detected smallest depth of each grid
                depth_colormap = cv2.putText(depth_colormap, str(round(x_obstacles[i],2)), (int(pxstep*(1/4 + i%3)),int(pystep*(1/3 + m.floor(i/3)))), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)                  
                # convert to body-frame coordinates in meters
                x,y,z = convert_depth_to_phys_coord_using_realsense(obstacle_list[i][1],obstacle_list[i][0],obstacle_list[i][2], 0)
            
            x_obstacles[i] = x # depth 
            y_obstacles[i] = y # height 
            z_obstacles[i] = z # width
            obstacle_vector.append([x,y,z])

        # Show Depth images
        cv2.namedWindow('Coloured DepthFrames',cv2.WINDOW_AUTOSIZE )
        cv2.imshow('Coloured DepthFrames', cv2.resize(depth_colormap, (int(DEPTH_WIDTH), int(DEPTH_HEIGHT))))
        
        key = cv2.waitKey(1)
        if (key == ord("v")):
            # show a vector plot of all the obstacles upon pressing "v". Good for debugging
            vector_plot(obstacle_vector)
               
finally:
    # Stop streaming
    pipe.stop()