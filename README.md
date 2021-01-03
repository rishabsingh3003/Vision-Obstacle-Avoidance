# Vision-Obstacle-Avoidance

This repository provides scripts to be used by onboard computers with Intel RealSense Camera's to provide 3-D obstacle avoidance features with ArduPilot. It also provides a easy to use script that supports grabbing depth camera data from Microsoft AirSim and sending Mavlink messages to ArduPilot SITL for simulation. 

## Objective:

The depth array as retrieved from the camera is divided into a 3x3 grid. The smallest depth in each grid is stored and converted into a body-frame NEU XYZ vector. This vector is packed into a mavlink message OBSTACLE_DISTANCE_3D (Supported locally by ArduPilot master ONLY) and sent to Flight Controller.

Currently the Avoidance system in ArduPilot is completely 2-D. This is being upgraded to 3D in the following PR: https://github.com/ArduPilot/ardupilot/pull/15553

![D455](https://i.ibb.co/XCSj9zZ/d455.png)


Demonstration of flight test with this Script and Intel RealSense D455: https://youtu.be/-6PWB52J3ho

Demonstration of AirSim: 
https://youtu.be/_G5LD6bPeJs


## Pre-Requirements for the entire project:

1. Clone the branch of the above mentioned PR.
2. Update pymavlink from ArduPilot repo.
3. ArduPilot Parameters (reboot after setting):
- PRX_TYPE = 2 (proximity lib will look for mavlink messages)
- AVOID_ENABLE = 3 (enable fence + proximity avoidance) 


## Companion Computer:

### Requirements: 

  - x86 based Companion Computer (for compatibility with Intel),
  Ubuntu 18.04 (otherwise, the following installation instruction might not work),
  - Python3 (default with Ubuntu 18.04)

### Install required packages: 

  - pip3 install pyrealsense2
  - pip3 install numpy
  - pip3 install pymavlink
  - pip3 install apscheduler
  - pip3 install pyserial
  - pip3 install numba           # Only necessary if you want to optimize the performance. - Require pip3 version >= 19 and llvmlite: pip3 install llvmlite==0.34.0
  - pip3 install opencv-python
  - sudo apt -y install python3-gst-1.0 gir1.2-gst-rtsp-server-1.0 gstreamer1 0-plugins-base gstreamer1.0-plugins-ugly libx264-dev

  - sudo apt install python3-opencv


### Steps to Launch:

1. Copy Companion_Computer directory onto the Companion Computer. Connect the camera.
3. Connect the computer to a flight controller running ArduPilot with Master + the above mentioned PR.
2. run script: d4xx_to_mavlink_3D.py --connect "Vehicle connection target string"

### Available arguments: 

- --connect: port/IP to connect to
- --baudrate: Set baudrate of connection 
- --obstacle_distance_msg_hz: Rate of sending mavlink message, defaults to 15hz
- --debug_enable: Displays RGB and depth view along with depth inside each grid (only enable if valid monitor is connected)
- --camera_name: Camera name to be connected to. If not specified, any valid camera will be connected to randomly. For eg: type 'D435I' to look for Intel RealSense D435I.

Once the camera is connected, check with Mission Planner if Obstacles are being read.


## AirSim:

The AirSim Script shares some of the code from the d4xx_to_mavlink script. The idea remains the same. Depth image is grabbed from AirSim and divided into a 3x3 grid.
This is a good way to experiment with depth cameras before moving onto real vehicle.

### Requirements:

1. Setup SITL: https://ardupilot.org/dev/docs/SITL-setup-landingpage.html
2. Setup SITL to work with AirSim: https://ardupilot.org/dev/docs/sitl-with-airsim.html

### Install required packages: 

  - pip3 install pymavlink
  - pip3 install apscheduler
  - pip3 install airsim
  - pip3 install numpy
  - sudo apt-get install python-PIL
  - sudo apt install python3-opencv

Add the following section to settings.json file that can be found in Documents/AirSim

```
"CameraDefaults": {
      "CaptureSettings": [
        {
          "ImageType": 1,
          "Width": 640,
          "Height": 480,
          "FOV_Degrees": 90,
          "AutoExposureSpeed": 100,
          "MotionBlurAmount": 0
        }
    ]
  },
```

### Steps to Launch:

1. Launch SITL: sim_vehicle.py -v ArduCopter -f airsim-copter --console --map
2. Launch AirSim: ./Blocks.sh -ResX=640 -ResY=480 -windowed
3. Launch Script: python3 air_sim_to_mavlink.py

### Available arguments: 

- --connect: port/IP to connect to
- --baudrate: Set baudrate of connection 
- --obstacle_distance_msg_hz: Rate of sending mavlink message, defaults to 15hz