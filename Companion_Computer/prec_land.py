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
sys.path.append("/home/rishabh/Vision-Obstacle-Avoidance/Companion_Computer/")
# Set MAVLink protocol to 2.
import os
os.environ["MAVLINK20"] = "1"
path = os.getcwd()

print(path)
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
import cv2.aruco as aruco
import random

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

obstacle_line_height_ratio = 0.18  # [0-1]: 0-Top, 1-Bottom. The height of the horizontal line to find distance to obstacle.
obstacle_line_thickness_pixel = 10 # [1-DEPTH_HEIGHT]: Number of pixel rows to use to generate the obstacle distance message. For each column, the scan will return the minimum value for those pixels centered vertically in the image.

USE_PRESET_FILE = False
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

# Use this to rotate all processed data
camera_facing_angle_degree = 0

# Store device serial numbers of connected camera
device_id = None

# Enable/disable each message/function individually
enable_msg_obstacle_distance = True
enable_msg_distance_sensor = False
obstacle_distance_msg_hz_default = 5.0
default_large_dist = 9999

# lock for thread synchronization
lock = threading.Lock()

# index for lower part of the depth frame in a 3x3 grid
lower_frame = [6,7,8]

mavlink_thread_should_exit = False

debug_enable_default = 1

# default exit code is failure - a graceful termination with a
# terminate signal is possible.
exit_code = 1

######################################################
##  Global variables                                ##
######################################################

# FCU connection variables
vehicle_pitch_rad = None

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
distances_array_length = 72
angle_offset = None
increment_f  = None
distances = np.ones((distances_array_length,), dtype=np.uint16) * (max_depth_cm + 1)

mavlink_obstacle_coordinates = np.ones((9,3), dtype = np.float) * (9999)
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
args = parser.parse_args()

connection_string = args.connect
connection_baudrate = args.baudrate
obstacle_distance_msg_hz = args.obstacle_distance_msg_hz
debug_enable = args.debug_enable
camera_name = args.camera_name

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

def send_obstacle_distance_3D_message(land_pos):
	global current_time_ms, mavlink_obstacle_coordinates
	global last_obstacle_distance_sent_ms
	
	if current_time_ms == last_obstacle_distance_sent_ms:
		# no new frame
		return
	last_obstacle_distance_sent_ms = current_time_ms
	x = -land_pos[0]
	z = land_pos[1]
	y = land_pos[2]
	# print (land_pos)
	x_offset_rad = m.atan(y / x)
	y_offset_rad = m.pi/2 - m.atan((np.sqrt(x * x + y * y))/z)
	print(m.degrees(y_offset_rad))
	distance = np.sqrt(x * x + y * y + z * z)
	conn.mav.landing_target_send(
		current_time_ms,                       # time target data was processed, as close to sensor capture as possible
		0,                                  # target num, not used
		mavutil.mavlink.MAV_FRAME_BODY_NED, # frame, not used
		x_offset_rad,                       # X-axis angular offset, in radians
		y_offset_rad,                       # Y-axis angular offset, in radians
		distance,                           # distance, in meters
		0,                                  # Target x-axis size, in radians
		0,                                  # Target y-axis size, in radians
		0,                                  # x	float	X Position of the landing target on MAV_FRAME
		0,                                  # y	float	Y Position of the landing target on MAV_FRAME
		0,                                  # z	float	Z Position of the landing target on MAV_FRAME
		(1,0,0,0),      # q	float[4]	Quaternion of landing target orientation (w, x, y, z order, zero-rotation is 1, 0, 0, 0)
		2,              # type of landing target: 2 = Fiducial marker
		1,              # position_valid boolean
	)

def send_msg_to_gcs(text_to_be_sent):
	return
	# MAV_SEVERITY: 0=EMERGENCY 1=ALERT 2=CRITICAL 3=ERROR, 4=WARNING, 5=NOTICE, 6=INFO, 7=DEBUG, 8=ENUM_END
	text_msg = 'D4xx: ' + text_to_be_sent
	conn.mav.statustext_send(mavutil.mavlink.MAV_SEVERITY_INFO, text_msg.encode())
	progress("INFO: %s" % text_to_be_sent)

# Request a timesync update from the flight controller, for future work.
# TODO: Inspect the usage of timesync_update 
def update_timesync(ts=0, tc=0):
	return
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


def convert_depth_to_phys_coord_using_realsense(depth_mat, depth_coordinates, depth):
	depth_img_width = depth_mat.shape[1]
	depth_img_height = depth_mat.shape[0]
	scale_x = depth_intrinsics.width/depth_img_width
	scale_y = depth_intrinsics.height/depth_img_height
	result = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [depth_coordinates[0] * scale_x, depth_coordinates[1] * scale_y], depth)

	center_pixel =  [depth_intrinsics.ppy, depth_intrinsics.ppx]
	result_center = rs.rs2_deproject_pixel_to_point(depth_intrinsics, center_pixel, depth)

	return -result[2], -(result[1] - result_center[1]), (result[0]- result_center[0])

def nothing(x):
	pass

def detect_shape(frame, depth_mat):
	# operations on the frame
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# set dictionary size depending on the aruco marker selected
	aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_50)

	# detector parameters can be set here (List of detection parameters[3])
	parameters = aruco.DetectorParameters_create()
	parameters.adaptiveThreshConstant = 10

	# lists of ids and the corners belonging to each id
	corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
	# frame = cv2.circle(frame,(300,400),10,(0,0,200),-1)
	# font for displaying text (below)
	font = cv2.FONT_HERSHEY_SIMPLEX

	# check if the ids list is not empty
	# if no check is added the code will crash
	if np.all(ids != None):
		# estimate pose of each marker and return the values
		# rvet and tvec-different from camera coefficients
		# rvec, tvec ,_ = aruco.estimatePoseSingleMarkers(corners, 0.05)
		# (rvec-tvec).any() # get rid of that nasty numpy value array error

		# for i in range(0, ids.size):
		#     # draw axis for the aruco markers
		#     aruco.drawAxis(frame, mtx, dist, rvec[i], tvec[i], 0.1)

		# draw a square around the markers
		aruco.drawDetectedMarkers(frame, corners)


		# code to show ids of the marker found
		strg = ''
		for i in range(0, ids.size):
			strg += str(ids[i][0])+', '

			x = np.round(((corners[i-1][0][0][0] + corners[i-1][0][1][0] + corners[i-1][0][2][0] + corners[i-1][0][3][0]) / 4),0)
			y = np.round(((corners[i-1][0][0][1] + corners[i-1][0][1][1] + corners[i-1][0][2][1] + corners[i-1][0][3][1]) / 4),0)

			# x1 = np.round((0.5*abs(corners[i-1][0][2][0]+corners[i-1][0][0][0])),0)
			# y1 = np.round((0.5*abs(corners[i-1][0][2][1]+corners[i-1][0][0][1])),0)

			# x2 = np.round((0.5*abs(corners[i-1][0][1][0]+corners[i-1][0][0][0])),0)
			# y2 = np.round((0.5*abs(corners[i-1][0][1][1]+corners[i-1][0][0][1])),0)

			cv2.putText(frame, "ID:" + str(ids[i-1][0]), (int(x-30),int(y-65)), font, 0.8, (255,0,0),2,cv2.LINE_AA)
			cv2.putText(frame, "," + str(y), (int(x),int(y-40)), font, 0.8, (0,0,255),2,cv2.LINE_AA)
			cv2.putText(frame, str(x), (int(x-80),int(y-40)), font, 0.8, (0,0,255),2,cv2.LINE_AA)
			# print (x,y)
			# rotM = np.zeros(shape=(3,3))
			# dst, jacobian = cv2.Rodrigues(rvec[i-1], rotM, jacobian = 0)

			# ypr = cv2.RQDecomp3x3(rotM)
			# yaw = np.round((ypr[0][0]),0)
			# cv2.putText(frame, "deg: " + str(angle), (0,250), font, 1, (0,255,0),2,cv2.LINE_AA)
			# val = (str(ids[i-1][0]), str(x),str(y), str(x2), str(y2))
			depth_img_width = depth_mat.shape[1]
			depth_img_height = depth_mat.shape[0]
			colour_img_width = frame.shape[1]
			colour_img_height = frame.shape[0]
			x_d = x / colour_img_width*depth_img_width
			y_d = y/ colour_img_height*depth_img_height
			# depth_mat = cv2.circle(depth_mat, (int(x_d),int(y_d)), 2, (255, 0, 0), 2)

			return int(x_d), int(y_d)

	else:
		# code to show 'No Ids' when no markers are found
		cv2.putText(frame, "No Ids", (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)
		return 999,999
		

	# # display the resulting frame
	# cv2.imshow('frame',frame)
	# cv2.waitKey(1)

def find_marker_location(depth_mat, depth_x, depth_y):
	original_depth_x = depth_x
	original_depth_y = depth_y
	for x in range(10):
		try:
			if (depth_x < depth_mat.shape[1] and depth_y < depth_mat.shape[0]):
				depth = depth_mat[int(depth_y), int(depth_x)] *depth_scale
				if (depth < DEPTH_RANGE_M[1] and depth > DEPTH_RANGE_M[0]):
					x,y,z = convert_depth_to_phys_coord_using_realsense(depth_mat, [int(depth_x), int(depth_y)], depth )
					return [x,y,z]
				depth_x = original_depth_x + random.randint(-5,5)
				depth_y = original_depth_y + random.randint(-5,5)
		except:
			continue



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


# # Send MAVlink messages in the background at pre-determined frequencies
sched = BackgroundScheduler()

if enable_msg_obstacle_distance:
	# sched.add_job(send_obstacle_distance_3D_message, 'interval', seconds = 1/obstacle_distance_msg_hz)
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

# sched.start()

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
		color_image = np.asanyarray(color_frame.get_data())

		x_land, y_land = detect_shape(color_image, depth_mat)
		land_pos = (find_marker_location(depth_mat, x_land, y_land))
		if (land_pos != None):
			send_obstacle_distance_3D_message(land_pos)
		if RTSP_STREAMING_ENABLE is True:
			color_image = np.asanyarray(color_frame.get_data())
			rtsp_streaming_img = color_image

		if debug_enable == 1:
			# Prepare the data
			input_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())
			output_image = np.asanyarray(colorizer.colorize(filtered_frame).get_data())
			color_image = np.asanyarray(color_frame.get_data())

			display_image = np.hstack((input_image, cv2.resize(output_image, (DEPTH_WIDTH, DEPTH_HEIGHT))))
			cv2.namedWindow('Coloured Frames', cv2.WINDOW_AUTOSIZE)
			cv2.imshow('Coloured Frames', cv2.resize(color_image, (int(DEPTH_WIDTH), int(DEPTH_HEIGHT))))

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
	# conn.close()
	progress("INFO: Realsense pipe and vehicle object closed.")
	sys.exit(exit_code)
