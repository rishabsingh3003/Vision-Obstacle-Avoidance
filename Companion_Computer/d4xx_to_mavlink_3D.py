#!/usr/bin/env python3

######################################################
##  librealsense D4xx to MAVLink                    ##
######################################################
# Requirements:
#   x86 based Companion Computer (for compatibility with Intel),
#   Ubuntu 18.04 (otherwise, the following installation instruction might not work),
#   Python3 (default with Ubuntu 18.04)
# Install required packages:
#   pip3 install pyrealsense2
#   pip3 install numpy
#   pip3 install pymavlink
#   pip3 install apscheduler
#   pip3 install pyserial
#   pip3 install numba           # Only necessary if you want to optimize the performance. Require pip3 version >= 19 and llvmlite: pip3 install llvmlite==0.34.0
#   pip3 install opencv-python
#   sudo apt -y install python3-gst-1.0 gir1.2-gst-rtsp-server-1.0 gstreamer1.0-plugins-base gstreamer1.0-plugins-ugly libx264-dev
# Only necessary if you installed the minimal version of Ubuntu:
#   sudo apt install python3-opencv

# Set the path for pyrealsense2.[].so
# Otherwise, place the pyrealsense2.[].so file under the same directory as this script or modify PYTHONPATH
import sys
sys.path.append("/usr/local/lib/")

# Set MAVLink protocol to 2.
import os
os.environ["MAVLINK20"] = "1"

# Import the libraries
import pyrealsense2 as rs
import numpy as np
import math as m
import signal
import sys
import time
import argparse
import threading
import json
from time import sleep
from apscheduler.schedulers.background import BackgroundScheduler
from pymavlink import mavutil
from numba import njit
import detect_land
import traceback

# In order to import cv2 under python3 when you also have ROS Kinetic installed
if os.path.exists("/opt/ros/kinetic/lib/python2.7/dist-packages"):
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') 
if os.path.exists("~/anaconda3/lib/python3.7/site-packages"):
    sys.path.append('~/anaconda3/lib/python3.7/site-packages')
import cv2

# To setup video streaming
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GLib

# To obtain ip address
import socket

######################################################
##  Depth parameters - reconfigurable               ##
######################################################

# Sensor-specific parameter, for D435: https://www.intelrealsense.com/depth-camera-d435/
STREAM_TYPE  = [rs.stream.depth, rs.stream.color]  # rs2_stream is a types of data provided by RealSense device
FORMAT       = [rs.format.z16, rs.format.bgr8]     # rs2_format is identifies how binary data is encoded within a frame
DEPTH_WIDTH  = 640               # Defines the number of columns for each frame or zero for auto resolve
DEPTH_HEIGHT = 480               # Defines the number of lines for each frame or zero for auto resolve
COLOR_WIDTH  = 640
COLOR_HEIGHT = 480
FPS          = 30
DEPTH_RANGE_M = [0.1, 10.0]       # Replace with your sensor's specifics, in meter

USE_PRESET_FILE = True
PRESET_FILE  = "presets/d4xx-high-accuracy.json"

RTSP_STREAMING_ENABLE = False
RTSP_PORT = "8554"
RTSP_MOUNT_POINT = "/d4xx"

# List of filters to be applied, in this order.
# https://github.com/IntelRealSense/librealsense/blob/master/doc/post-processing-filters.md

filters = [
    [True,  "Decimation Filter",   rs.decimation_filter(2)],
    [True,  "Threshold Filter",    rs.threshold_filter()],
    [True,  "Depth to Disparity",  rs.disparity_transform(True)],
    [True,  "Spatial Filter",      rs.spatial_filter()],
    [True,  "Temporal Filter",     rs.temporal_filter()],
    [False, "Hole Filling Filter", rs.hole_filling_filter()],
    [True,  "Disparity to Depth",  rs.disparity_transform(False)]
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
connection_string_default = 'localhost:14551'
connection_baudrate_default = 921600

# Store device serial numbers of connected camera
device_id = None

# Enable/disable each message/function individually
enable_msg_obstacle_distance = True
obstacle_distance_msg_hz_default = 15.0
default_large_dist = 9999

# lock for thread synchronization
lock = threading.Lock()

mavlink_thread_should_exit = False

debug_enable_default = 1

ground_detection_default = 1

# default exit code is failure - a graceful termination with a
# terminate signal is possible.
exit_code = 1

######################################################
##  Global variables                                ##
######################################################

# Camera-related variables
pipe = None
depth_scale = 0
colorizer = rs.colorizer()
depth_hfov_deg = None
depth_vfov_deg = None
depth_intrinsics = None

# The name of the display window
display_name  = 'Input/output depth'
rtsp_streaming_img = None

# Data variables
data = None
current_time_us = 0

start_time =  int(round(time.time() * 1000))
current_milli_time = lambda: int(round(time.time() * 1000) - start_time)

current_time_ms = current_milli_time()
last_obstacle_distance_sent_ms = 0  # value of current_time_us when obstacle_distance last sent

# Obstacle distances in front of the sensor, starting from the left in increment degrees to the right
# See here: https://mavlink.io/en/messages/common.html#OBSTACLE_DISTANCE
min_depth_cm = int(DEPTH_RANGE_M[0] * 100)  # In cm
max_depth_cm = int(DEPTH_RANGE_M[1] * 100)  # In cm, should be a little conservative

mavlink_obstacle_coordinates = np.ones((9,3), dtype = np.float) * (9999)

# part of the frame to use ground detection. This is a number between 0-1.
ground_detection_frame_size = 0.5
# obstacles closer that this many meters to the ground will be ignored
ground_detection_thresh = 0.15

# sampling step in x
step_size_x = 8
# sampling step in y
step_size_y = 8
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
parser.add_argument('--camera_name', type=str,
                    help="Camera name to be connected to. If not specified, any valid camera will be connected to randomly. For eg: type 'D435I' to look for Intel RealSense D435I.")
parser.add_argument('--ground_detection_enabled',type=float,
                    help="Enable ground detection. This needs Numba to be installed.")
args = parser.parse_args()

connection_string = args.connect
connection_baudrate = args.baudrate
obstacle_distance_msg_hz = args.obstacle_distance_msg_hz
debug_enable = args.debug_enable
camera_name = args.camera_name
ground_detection_enable = args.ground_detection_enabled

def progress(string):
    print(string, file=sys.stdout)
    sys.stdout.flush()

# Using default values if no specified inputs
if not connection_string:
    connection_string = connection_string_default
    progress("INFO: Using default connection_string %s" % connection_string)
else:
    progress("INFO: Using connection_string %s" % connection_string)

if not connection_baudrate:
    connection_baudrate = connection_baudrate_default
    progress("INFO: Using default connection_baudrate %s" % connection_baudrate)
else:
    progress("INFO: Using connection_baudrate %s" % connection_baudrate)

if not obstacle_distance_msg_hz:
    obstacle_distance_msg_hz = obstacle_distance_msg_hz_default
    progress("INFO: Using default obstacle_distance_msg_hz %s" % obstacle_distance_msg_hz)
else:
    progress("INFO: Using obstacle_distance_msg_hz %s" % obstacle_distance_msg_hz)

# The list of filters to be applied on the depth image
for i in range(len(filters)):
    if filters[i][0] is True:
        progress("INFO: Applying: %s" % filters[i][1])
    else:
        progress("INFO: NOT applying: %s" % filters[i][1])

if not debug_enable:
    debug_enable = debug_enable_default

if not ground_detection_enable:
    ground_detection_enable =  ground_detection_default

if debug_enable == 1:
    progress("INFO: Debugging option enabled")
    cv2.namedWindow(display_name, cv2.WINDOW_AUTOSIZE)
else:
    progress("INFO: Debugging option DISABLED")

######################################################
##  Functions - MAVLink                             ##
######################################################

def mavlink_loop(conn, callbacks):
    '''a main routine for a thread; reads data from a mavlink connection,
    calling callbacks based on message type received.
    '''
    interesting_messages = list(callbacks.keys())
    while not mavlink_thread_should_exit:
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

def send_obstacle_distance_3D_message():
    global current_time_ms, mavlink_obstacle_coordinates
    global last_obstacle_distance_sent_ms

    if current_time_ms == last_obstacle_distance_sent_ms:
        # no new frame
        return
    last_obstacle_distance_sent_ms = current_time_ms

    # ArduPilot has a easier time if you sort the array with distance. This is totally optional though. Might save a little bit CPU on the flight controller
    sorted_array = []
    for i in range(9):
        dist = pow(mavlink_obstacle_coordinates[i][0],2) + pow(mavlink_obstacle_coordinates[i][1],2) + pow(mavlink_obstacle_coordinates[i][2],2)
        sorted_array.append([dist,[mavlink_obstacle_coordinates[i][0], mavlink_obstacle_coordinates[i][1], mavlink_obstacle_coordinates[i][2]]])

    for i in range(9):
        conn.mav.obstacle_distance_3d_send(
            current_time_ms,    # us Timestamp (UNIX time or time since system boot)
            0,
            mavutil.mavlink.MAV_FRAME_BODY_FRD,
            65535,
            float(sorted_array[i][1][0]),
            float(sorted_array[i][1][1]),
            float(sorted_array[i][1][2]),
            float(DEPTH_RANGE_M[0]),
            float(DEPTH_RANGE_M[1])
        )

def send_msg_to_gcs(text_to_be_sent):
    # MAV_SEVERITY: 0=EMERGENCY 1=ALERT 2=CRITICAL 3=ERROR, 4=WARNING, 5=NOTICE, 6=INFO, 7=DEBUG, 8=ENUM_END
    text_msg = 'D4xx: ' + text_to_be_sent
    conn.mav.statustext_send(mavutil.mavlink.MAV_SEVERITY_INFO, text_msg.encode())
    progress("INFO: %s" % text_to_be_sent)

# Request a timesync update from the flight controller, for future work.
# TODO: Inspect the usage of timesync_update
def update_timesync(ts=0, tc=0):
    if ts == 0:
        ts = int(round(time.time() * 1000))
    conn.mav.timesync_send(tc, ts)


######################################################
##  Functions - D4xx cameras                        ##
######################################################

DS5_product_ids = ["0AD1", "0AD2", "0AD3", "0AD4", "0AD5", "0AF6", "0AFE", "0AFF", "0B00", "0B01", "0B03", "0B07","0B3A", "0B5C"]

def find_device_that_supports_advanced_mode() :
    global device_id
    ctx = rs.context()
    ds5_dev = rs.device()
    devices = ctx.query_devices()
    for dev in devices:
        if dev.supports(rs.camera_info.product_id) and str(dev.get_info(rs.camera_info.product_id)) in DS5_product_ids:
            name = rs.camera_info.name
            if dev.supports(name):
                if not camera_name or (camera_name.lower() == dev.get_info(name).split()[2].lower()):
                    progress("INFO: Found device that supports advanced mode: %s" % dev.get_info(name))
                    device_id = dev.get_info(rs.camera_info.serial_number)
                    return dev
    raise Exception("No device that supports advanced mode was found")

# Loop until we successfully enable advanced mode
def realsense_enable_advanced_mode(advnc_mode):
    while not advnc_mode.is_enabled():
        progress("INFO: Trying to enable advanced mode...")
        advnc_mode.toggle_advanced_mode(True)
        # At this point the device will disconnect and re-connect.
        progress("INFO: Sleeping for 5 seconds...")
        time.sleep(5)
        # The 'dev' object will become invalid and we need to initialize it again
        dev = find_device_that_supports_advanced_mode()
        advnc_mode = rs.rs400_advanced_mode(dev)
        progress("INFO: Advanced mode is %s" "enabled" if advnc_mode.is_enabled() else "disabled")

# Load the settings stored in the JSON file
def realsense_load_settings_file(advnc_mode, setting_file):
    # Sanity checks
    if os.path.isfile(setting_file):
        progress("INFO: Setting file found %s" % setting_file)
    else:
        progress("INFO: Cannot find setting file %s" % setting_file)
        exit()

    if advnc_mode.is_enabled():
        progress("INFO: Advanced mode is enabled")
    else:
        progress("INFO: Device does not support advanced mode")
        exit()

    # Input for load_json() is the content of the json file, not the file path
    with open(setting_file, 'r') as file:
        json_text = file.read().strip()

    advnc_mode.load_json(json_text)

# Establish connection to the Realsense camera
def realsense_connect():
    global pipe, depth_scale, depth_intrinsics
    # Declare RealSense pipe, encapsulating the actual device and sensors
    pipe = rs.pipeline()

    # Configure image stream(s)
    config = rs.config()
    if device_id:
        # connect to a specific device ID
        config.enable_device(device_id)
    config.enable_stream(STREAM_TYPE[0], DEPTH_WIDTH, DEPTH_HEIGHT, FORMAT[0], FPS)
    config.enable_stream(STREAM_TYPE[1], COLOR_WIDTH, COLOR_HEIGHT, FORMAT[1], FPS)

    # Start streaming with requested config
    profile = pipe.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    depth_intrinsics = profile.get_stream(STREAM_TYPE[0]).as_video_stream_profile().intrinsics

    progress("INFO: Depth scale is: %s" % depth_scale)

def realsense_configure_setting(setting_file):
    device = find_device_that_supports_advanced_mode()
    advnc_mode = rs.rs400_advanced_mode(device)
    realsense_enable_advanced_mode(advnc_mode)
    realsense_load_settings_file(advnc_mode, setting_file)

# convert depth to FRD body-frame vector
def convert_depth_to_phys_coord_using_realsense(depth_coordinates, depth):
  result = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [depth_coordinates[0], depth_coordinates[1]], depth)
  center_pixel =  [depth_intrinsics.ppy/2,depth_intrinsics.ppx/2]
  result_center = rs.rs2_deproject_pixel_to_point(depth_intrinsics, center_pixel, depth)
  return result[2], (result[1] - result_center[1]), (result[0]- result_center[0])

# Samples points on the frame in pre-determined steps. Converts those points into a point cloud.
def segment_frame(depth_mat):
    # Parameters for depth image
    depth_img_width  = depth_mat.shape[1]
    depth_img_height = depth_mat.shape[0]

    reduced_depth_mat = np.zeros(depth_mat.shape)
    reduced_pc = np.zeros((depth_img_height, depth_img_width, 3), dtype= np.float)

    bottom_frame_considered_points = 0
    bottom_frame_valid_points = 0
    half_frame_height = int(0.5 * depth_img_height)

    for y_pixel in range(0,depth_img_height, step_size_y):
        for x_pixel in range(0, depth_img_width, step_size_x):
            if (y_pixel > half_frame_height):
                bottom_frame_considered_points = bottom_frame_considered_points + 1
            point_depth = depth_mat[y_pixel,x_pixel] * depth_scale
            if point_depth > DEPTH_RANGE_M[0] and point_depth < DEPTH_RANGE_M[1]:
                if (y_pixel > half_frame_height):
                    bottom_frame_valid_points = bottom_frame_valid_points + 1
                reduced_depth_mat[y_pixel, x_pixel] = point_depth
                body_x, body_y, body_z = convert_depth_to_phys_coord_using_realsense([y_pixel,x_pixel], point_depth)
                reduced_pc[y_pixel, x_pixel] = [body_x, body_y, body_z]

    bottom_frame_healthy = True
    if (bottom_frame_considered_points == 0):
        bottom_frame_healthy = False
    else:
        pct_valid_points = bottom_frame_valid_points/bottom_frame_considered_points
        if pct_valid_points < 0.50:
            bottom_frame_healthy = False

    if not bottom_frame_healthy:
        #less than half of the bottom frame is available. Ground detection will not be possible. Lets remove it.
        reduced_depth_mat[half_frame_height:depth_img_height, 0:depth_img_width] = 0
        depth_mat[half_frame_height:depth_img_height, 0:depth_img_width] = 0
        reduced_pc[half_frame_height:depth_img_height, 0:depth_img_width] = [0, 0, 0]

    return reduced_depth_mat, reduced_pc, bottom_frame_healthy

# divides the frame into a 3x3 grid and picks out the closest obstacle in each grid
@njit
def grid_distances(depth_mat, reduced_depth_mat, reduced_pc, obstacle_distance_list, obstacle_vector_list, ground_eqn, valid_eqn):
    # Parameters for depth image
    depth_img_width  = depth_mat.shape[1]
    depth_img_height = depth_mat.shape[0]
    sampling_width = int(1/3 * depth_img_width)
    sampling_height = int(1/3* depth_img_height)

    for i in range(9):
        if i%3 == 0 and i != 0:
            sampling_width = int(1/3* depth_img_width)
            sampling_height = sampling_height + int(1/3 * depth_img_height)

        x_pixel = sampling_width - int(1/3 * depth_img_width)
        # print(i, sampling_width, sampling_height)
        while x_pixel < sampling_width:
            y_pixel = sampling_height - int(1/3* depth_img_height)
            while y_pixel < sampling_height:
                obs_vector = reduced_pc[y_pixel, x_pixel]
                if (obs_vector[0]):
                    dist_to_obs = np.linalg.norm(obs_vector)
                    if (dist_to_obs < obstacle_distance_list[i]):
                        if (valid_eqn):
                            distance_to_plane = detect_land.distance_plane_to_point(ground_eqn, obs_vector)
                            if (distance_to_plane > ground_detection_thresh):
                                obstacle_distance_list[i] = dist_to_obs
                                obstacle_vector_list[i] = obs_vector
                        else:
                            obstacle_distance_list[i] = dist_to_obs
                            obstacle_vector_list[i] = obs_vector
                y_pixel = y_pixel + 1
            x_pixel = x_pixel + 1
        sampling_width = sampling_width + int(1/3* depth_img_width)

# puts circle on the frame where ground was detected
def filter_ground(depth_mat, reduced_depth_mat, pc, land_detect_area, ground_eqn):
    width = pc.shape[0]
    height = pc.shape[1]
    for u in range(int(width*land_detect_area), width, step_size_x):
            for v in range(0, height, step_size_y):
                point = pc[u, v]
                distance_to_plane = detect_land.distance_plane_to_point(ground_eqn, point)
                if (distance_to_plane < ground_detection_thresh):
                    cv2.circle(depth_mat, (v, u), 4, color = (0,0, 0), thickness = 4)

######################################################
##  Functions - RTSP Streaming                      ##
##  Adapted from https://github.com/VimDrones/realsense-helper/blob/master/fisheye_stream_to_rtsp.py, credit to: @Huibean (GitHub)
######################################################

class SensorFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self, **properties):
        super(SensorFactory, self).__init__(**properties)
        self.number_frames = 0
        self.fps = FPS
        self.duration = 1 / self.fps * Gst.SECOND
        self.launch_string = 'appsrc name=source is-live=true block=true format=GST_FORMAT_TIME ' \
                             'caps=video/x-raw,format=BGR,width={},height={},framerate={}/1 ' \
                             '! videoconvert ! video/x-raw,format=I420 ' \
                             '! x264enc speed-preset=ultrafast tune=zerolatency ' \
                             '! rtph264pay config-interval=1 name=pay0 pt=96'.format(COLOR_WIDTH, COLOR_HEIGHT, self.fps)

    def on_need_data(self, src, length):
        global rtsp_streaming_img
        frame = rtsp_streaming_img
        if frame is not None:
            data = frame.tobytes()
            buf = Gst.Buffer.new_allocate(None, len(data), None)
            buf.fill(0, data)
            buf.duration = self.duration
            timestamp = self.number_frames * self.duration
            buf.pts = buf.dts = int(timestamp)
            buf.offset = timestamp
            self.number_frames += 1
            retval = src.emit('push-buffer', buf)
            if retval != Gst.FlowReturn.OK:
                progress(retval)

    def do_create_element(self, url):
        return Gst.parse_launch(self.launch_string)

    def do_configure(self, rtsp_media):
        self.number_frames = 0
        appsrc = rtsp_media.get_element().get_child_by_name('source')
        appsrc.connect('need-data', self.on_need_data)


class GstServer(GstRtspServer.RTSPServer):
    def __init__(self, **properties):
        super(GstServer, self).__init__(**properties)
        factory = SensorFactory()
        factory.set_shared(True)
        self.get_mount_points().add_factory(RTSP_MOUNT_POINT, factory)
        self.attach(None)

def get_local_ip():
    local_ip_address = "127.0.0.1"
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 1))  # connect() for UDP doesn't send packets
        local_ip_address = s.getsockname()[0]
    except:
        local_ip_address = socket.gethostbyname(socket.gethostname())
    return local_ip_address


######################################################
##  Main code starts here                           ##
######################################################

try:
    # Note: 'version' attribute is supported from pyrealsense2 2.31 onwards and might require building from source
    progress("INFO: pyrealsense2 version: %s" % str(rs.__version__))
except Exception:
    # fail silently
    pass

progress("INFO: Starting Vehicle communications")
print (connection_string)
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
# connecting and configuring the camera is a little hit-and-miss.
# Start a timer and rely on a restart of the script to get it working.
# Configuring the camera appears to block all threads, so we can't do
# this internally.

# send_msg_to_gcs('Setting timer...')
signal.setitimer(signal.ITIMER_REAL, 5)  # seconds...

send_msg_to_gcs('Connecting to camera...')
if USE_PRESET_FILE:
    realsense_configure_setting(PRESET_FILE)
realsense_connect()
send_msg_to_gcs('Camera connected.')

signal.setitimer(signal.ITIMER_REAL, 0)  # cancel alarm


# Send MAVlink messages in the background at pre-determined frequencies
sched = BackgroundScheduler()

if enable_msg_obstacle_distance:
    sched.add_job(send_obstacle_distance_3D_message, 'interval', seconds = 1/obstacle_distance_msg_hz)
    send_msg_to_gcs('Sending obstacle distance messages to FCU')
else:
    send_msg_to_gcs('Nothing to do. Check params to enable something')
    pipe.stop()
    conn.mav.close()
    progress("INFO: Realsense pipe and vehicle object closed.")
    sys.exit()

glib_loop = None
if RTSP_STREAMING_ENABLE is True:
    send_msg_to_gcs('RTSP at rtsp://' + get_local_ip() + ':' + RTSP_PORT + RTSP_MOUNT_POINT)
    Gst.init(None)
    server = GstServer()
    glib_loop = GLib.MainLoop()
    glib_thread = threading.Thread(target=glib_loop.run, args=())
    glib_thread.start()
else:
    send_msg_to_gcs('RTSP not streaming')

sched.start()

# gracefully terminate the script if an interrupt signal (e.g. ctrl-c)
# is received.  This is considered to be abnormal termination.
main_loop_should_quit = False
def sigint_handler(sig, frame):
    global main_loop_should_quit
    main_loop_should_quit = True
signal.signal(signal.SIGINT, sigint_handler)

# gracefully terminate the script if a terminate signal is received
# (e.g. kill -TERM).
def sigterm_handler(sig, frame):
    global main_loop_should_quit
    main_loop_should_quit = True
    global exit_code
    exit_code = 0

signal.signal(signal.SIGTERM, sigterm_handler)

# Begin of the main loop
last_time = time.time()
try:
    while not main_loop_should_quit:
        # This call waits until a new coherent set of frames is available on a device
        # Calls to get_frame_data(...) and get_frame_timestamp(...) on a device will return stable values until wait_for_frames(...) is called
        frames = pipe.wait_for_frames()
        align = rs.align(rs.stream.color)
        frames = align.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Store the timestamp for MAVLink messages
        current_time_ms = current_milli_time()

        # Apply the filters
        filtered_frame = depth_frame
        for i in range(len(filters)):
            if filters[i][0] is True:
                filtered_frame = filters[i][2].process(filtered_frame)

        # Extract depth in matrix form
        depth_data = filtered_frame.as_frame().get_data()
        depth_mat = np.asanyarray(depth_data)

        reduced_depth_mat, reduced_pc, bottom_frame_healthy = segment_frame(depth_mat)

        if (ground_detection_enable and bottom_frame_healthy):
            plane_eqn, valid_eqn = detect_land.run(reduced_pc, ground_detection_frame_size, step_size_x, step_size_y)
        else:
            plane_eqn, valid_eqn = np.array([0.0, 0.0, 0.0, 0.0]), False
        # Create obstacle distance data from depth image
        depth_list = np.ones((9,), dtype=np.float) * (DEPTH_RANGE_M[1] + 1)
        obstacle_list = np.ones((9,2), dtype=np.uint16) * (default_large_dist)
        obstacle_vector_list = np.ones((9,3), dtype=np.float) * (default_large_dist)
        grid_distances(depth_mat, reduced_depth_mat, reduced_pc, depth_list, obstacle_vector_list, plane_eqn, valid_eqn)

        mavlink_obstacle_coordinates = obstacle_vector_list

        if RTSP_STREAMING_ENABLE is True:
            color_image = np.asanyarray(color_frame.get_data())
            rtsp_streaming_img = color_image

        if debug_enable == 1:
            # Prepare the data
            if valid_eqn:
                filter_ground(depth_mat, reduced_depth_mat, reduced_pc, ground_detection_frame_size, plane_eqn)

            input_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())
            output_image = np.asanyarray(colorizer.colorize(filtered_frame).get_data())
            color_image = np.asanyarray(color_frame.get_data())

                # divide view into 3x3 matrix
            pxstep = int(depth_mat.shape[1]/3)
            pystep = int(depth_mat.shape[0]/3)
            gx = pxstep
            gy = pystep
            while gx < depth_mat.shape[1]:
                cv2.line(output_image, (gx, 0), (gx, depth_mat.shape[0]), color=(0, 0, 0), thickness=1)
                gx += pxstep
            while gy < depth_mat.shape[0]:
                cv2.line(output_image, (0, gy), (depth_mat.shape[1], gy), color=(0, 0, 0),thickness=1)
                gy += pystep

            for i in range(len(depth_list)):
                # output_image = cv2.putText(output_image, str(round(depth_list[i],2)), (int(pxstep*(1/4 + i%3)),int(pystep*(1/3 + m.floor(i/3)))), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)
                if (depth_list[i] < DEPTH_RANGE_M[1]):
                    output_image = cv2.putText(output_image, str(round(obstacle_vector_list[i][0],2)), (int(pxstep*(1/4 + i%3)),int(pystep*(1/3 + m.floor(i/3)))), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)

            display_image = np.hstack((input_image, cv2.resize(output_image, (DEPTH_WIDTH, DEPTH_HEIGHT))))
            cv2.namedWindow('Coloured Frames', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Coloured Frames', cv2.resize(color_image, (int(DEPTH_WIDTH), int(DEPTH_HEIGHT))))

            # cv2.namedWindow('test', cv2.WINDOW_AUTOSIZE)
            # cv2.imshow('test', cv2.resize(reduced_depth_mat, (int(DEPTH_WIDTH), int(DEPTH_HEIGHT))))

            # Put the fps in the corner of the image
            processing_speed = 1 / (time.time() - last_time)
            text = ("%0.2f" % (processing_speed,)) + ' fps'
            textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            cv2.putText(display_image,
                        text,
                        org = (int((display_image.shape[1] - textsize[0]/2)), int((textsize[1])/2)),
                        fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale = 0.5,
                        thickness = 1,
                        color = (255, 255, 255))

            # Show the images
            cv2.imshow(display_name, display_image)
            cv2.waitKey(1)

            last_time = time.time()

except Exception as e:
    print (e)
    progress(e)
    print(traceback.format_exc())

except:
    send_msg_to_gcs('ERROR: Depth camera disconnected')

finally:
    progress('Closing the script...')
    # start a timer in case stopping everything nicely doesn't work.
    signal.setitimer(signal.ITIMER_REAL, 5)  # seconds...
    if glib_loop is not None:
        glib_loop.quit()
        glib_thread.join()
    pipe.stop()
    mavlink_thread_should_exit = True
    conn.close()
    progress("INFO: Realsense pipe and vehicle object closed.")
    sys.exit(exit_code)
