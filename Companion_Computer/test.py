# # OpenCV program to perform Edge detection in real time 
# # import libraries of python OpenCV  
# # where its functionality resides 
# import cv2  
  
# # np is an alias pointing to numpy library 
# import numpy as np 
  
  
# # capture frames from a camera 
# cap = cv2.VideoCapture(0) 
  
  
# # loop runs if capturing has been initialized 
# while(1): 
  
#     # reads frames from a camera 
#     ret, frame = cap.read() 
  
#     # converting BGR to HSV 
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
      
#     # define range of red color in HSV 
#     lower_red = np.array([30,150,50]) 
#     upper_red = np.array([255,255,180]) 
      
#     # create a red HSV colour boundary and  
#     # threshold HSV image 
#     mask = cv2.inRange(hsv, lower_red, upper_red) 
  
#     # Bitwise-AND mask and original image 
#     res = cv2.bitwise_and(frame,frame, mask= mask) 
  
#     # Display an original image 
#     cv2.imshow('Original',frame) 
  
#     # finds edges in the input image image and 
#     # marks them in the output map edges 
#     edges = cv2.Canny(frame,100,200) 
  
#     # Display edges in a frame 
#     cv2.imshow('Edges',edges) 
  
#     # Wait for Esc key to stop 
#     k = cv2.waitKey(5) & 0xFF
#     if k == 27: 
#         break
  
  
# # Close the window 
# cap.release() 
  
# # De-allocate any associated memory usage 
# cv2.destroyAllWindows()  

import datetime as dt
import socket
from numpy import *
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg


### START QtApp #####
app = QtGui.QApplication([])            # you MUST do this once (initialize things)
####################

win_uf = pg.GraphicsWindow(title="Signal from Depth Camera") # creates a window
p_uf = win_uf.addPlot(title="Input Velocity")  # creates empty space for the plot in the window
# p_uf.setYRange(0,8,padding=0)
curve_uf = p_uf.plot()                        # create an empty "plot" (a curve to plot)   

p_f = win_uf.addPlot(title="Output Velocity")  # creates empty space for the plot in the window
# p_f.setYRange(0,8,padding=0)
curve_f = p_f.plot()                        # create an empty "plot" (a curve to plot)


a_uf = win_uf.addPlot(title="Expected Acceleration")  # creates empty space for the plot in the window
# a_uf.setYRange(0,8,padding=0)
curve_auf = a_uf.plot()  


# a_f = win_uf.addPlot(title="Acceleration Limited")  # creates empty space for the plot in the window
# a_f.setYRange(-10,10,padding=0)
# curve_af = a_f.plot()  

windowWidth = 500                       # width of the window displaying the curve
Xm_uf = linspace(0,0,windowWidth)          # create array that will contain the relevant time series     
ptr_uf = -windowWidth                      # set first x position
Xm_f = linspace(0,0,windowWidth)          # create array that will contain the relevant time series     
ptr_f = -windowWidth                      # set first x position

Xm_auf = linspace(0,0,windowWidth)          # create array that will contain the relevant time series     
ptr_auf = -windowWidth                      # set first x position
Xm_af = linspace(0,0,windowWidth)          # create array that will contain the relevant time series     
ptr_af = -windowWidth                      # set first x position



def get_data():
    localIP     = "127.0.0.1"
    localPort   = 9002
    bufferSize  = 1024

    # Create a datagram socket
    UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

    # Bind to address and ip
    UDPServerSocket.bind((localIP, localPort))
    UDPServerSocket.setblocking(0)	
    while True:
        try: 
            bytesAddressPair = UDPServerSocket.recvfrom(bufferSize)
            message = bytesAddressPair[0]
            message = message.decode('utf-8')
            message.replace(" ", "")
            arr = message.split("#")
            print (arr)
            return arr[0],arr[1], arr[2]
        except:
            continue


# Realtime data plot. Each time this function is called, the data display is updated
def update():
    global curve_uf, ptr_uf, Xm_uf, curve_f, ptr_f, Xm_f, curve_auf, ptr_auf, Xm_auf, curve_af, ptr_af, Xm_af
    Xm_uf [:-1] = Xm_uf [1:]                      # shift data in the temporal mean 1 sample left
    value, value2, value3 = get_data()                   # read line (single value) from the serial port
    Xm_uf [-1] = float(value)                 # vector containing the instantaneous values      
    ptr_uf  += 1                              # update x position for displaying the curve
    curve_uf.setData(Xm_uf )                     # set the curve with this data
    curve_uf.setPos(ptr_uf,0)                   # set x position in the graph to 0

    Xm_f [:-1] = Xm_f [1:]                      # shift data in the temporal mean 1 sample left                  # read line (single value) from the serial port
    Xm_f [-1] = float(value2)                 # vector containing the instantaneous values      
    ptr_f  += 1                              # update x position for displaying the curve
    curve_f.setData(Xm_f )                     # set the curve with this data
    curve_f.setPos(ptr_f,0)                   # set x position in the graph to 0

    # Xm_af [:-1] = Xm_af [1:]                      # shift data in the temporal mean 1 sample left                  # read line (single value) from the serial port
    # Xm_af [-1] = float(value4)                 # vector containing the instantaneous values      
    # ptr_af  += 1                              # update x position for displaying the curve
    # curve_af.setData(Xm_af )                     # set the curve with this data
    # curve_af.setPos(ptr_af,0)                   # set x position in the graph to 0

    Xm_auf [:-1] = Xm_auf [1:]                      # shift data in the temporal mean 1 sample left                  # read line (single value) from the serial port
    Xm_auf [-1] = float(value3)                 # vector containing the instantaneous values      
    ptr_auf  += 1                              # update x position for displaying the curve
    curve_auf.setData(Xm_auf )                     # set the curve with this data
    curve_auf.setPos(ptr_auf,0)                   # set x position in the graph to 0
    
    
    QtGui.QApplication.processEvents()    # you MUST process the plot now

### MAIN PROGRAM #####    
# this is a brutal infinite loop calling your realtime data plot
while True: update()

### END QtApp ####
pg.QtGui.QApplication.exec_() # you MUST put this at the end
##################
