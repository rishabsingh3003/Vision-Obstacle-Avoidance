import socket
from apscheduler.schedulers.background import BackgroundScheduler
from pymavlink import mavutil
import threading
import time
import numpy as np
import pickle
import sys


UDP_IP = "127.0.0.1"
UDP_PORT_CAM1 = 5005
UDP_PORT_CAM2 = 6005

sock_cam1 = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP
sock_cam1.bind((UDP_IP, UDP_PORT_CAM1))

sock_cam2 = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP
sock_cam2.bind((UDP_IP, UDP_PORT_CAM2))

start_time =  int(round(time.time() * 1000))
current_milli_time = lambda: int(round(time.time() * 1000) - start_time)

mavlink_obstacle_coordinates_cam1 = np.ones((9, 3), dtype=np.float) * 9999
mavlink_obstacle_coordinates_cam2 = np.ones((9, 3), dtype=np.float) * 9999
DEPTH_RANGE_M = [0.1, 15]

# Default configurations for connection to the FCU
connection_string = 'localhost:14551'
connection_baudrate = 921600

obstacle_distance_msg_hz = 10


######################################################
##  Functions - MAVLink                             ##
######################################################

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

def send_obstacle_distance_3D_message():
    global mavlink_obstacle_coordinates_cam1, mavlink_obstacle_coordinates_cam2
    global last_obstacle_distance_sent_ms

    current_time_ms = current_milli_time()

    for i in range(9):
        conn.mav.obstacle_distance_3d_send(
            current_time_ms,    # us Timestamp (UNIX time or time since system boot)
            0,
            mavutil.mavlink.MAV_FRAME_BODY_FRD,
            65535,
            float(mavlink_obstacle_coordinates_cam1[i][0]),
            float(mavlink_obstacle_coordinates_cam1[i][1]),
            float(mavlink_obstacle_coordinates_cam1[i][2]),
            float(DEPTH_RANGE_M[0]),
            float(DEPTH_RANGE_M[1])
        )
        conn.mav.obstacle_distance_3d_send(
            current_time_ms,    # us Timestamp (UNIX time or time since system boot)
            0,
            mavutil.mavlink.MAV_FRAME_BODY_FRD,
            65535,
            float(mavlink_obstacle_coordinates_cam2[i][0]),
            float(mavlink_obstacle_coordinates_cam2[i][1]),
            float(mavlink_obstacle_coordinates_cam2[i][2]),
            float(DEPTH_RANGE_M[0]),
            float(DEPTH_RANGE_M[1])
        )

def send_msg_to_gcs(text_to_be_sent):
    # MAV_SEVERITY: 0=EMERGENCY 1=ALERT 2=CRITICAL 3=ERROR, 4=WARNING, 5=NOTICE, 6=INFO, 7=DEBUG, 8=ENUM_END
    text_msg = 'D4xx: ' + text_to_be_sent
    conn.mav.statustext_send(mavutil.mavlink.MAV_SEVERITY_INFO, text_msg.encode())

# Request a timesync update from the flight controller, for future work.
# TODO: Inspect the usage of timesync_update 
def update_timesync(ts=0, tc=0):
    if ts == 0:
        ts = int(round(time.time() * 1000))
    conn.mav.timesync_send(tc, ts)




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
send_msg_to_gcs('Sending obstacle distance messages to FCU')


sched.start()

if __name__ == '__main__':
    while True:
        try:
            data, addr = sock_cam1.recvfrom(1024)
            L = pickle.loads(data)
            mavlink_obstacle_coordinates_cam1 = L

            data, addr = sock_cam2.recvfrom(1024)
            L = pickle.loads(data)
            mavlink_obstacle_coordinates_cam2 = L

            time.delay(0.05)
        except KeyboardInterrupt:
            sys.exit()
        except:
            continue