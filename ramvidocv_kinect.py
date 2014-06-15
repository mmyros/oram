#%%
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
import pylab
from freenect import sync_get_depth as get_depth, sync_get_video as get_video
##### RECORDED VIDEO: ######
#cap = cv2.VideoCapture('/home/m/Dropbox/maze/video/examples/3.mp4')
##### CAPTURE    ########
#cap = cv2.VideoCapture(1)
#cap = cv2.cvCaptureFromCAM()

#fgbg = cv2.BackgroundSubtractorMOG2()
##%%
### here be CAMSHIFT
#
## take first frame of the video
#ret,frame = cap.read()
#print(frame)
## setup initial location of window
#r,h,c,w = 250,90,400,125  # simply hardcoded the values
#track_window = (c,r,w,h)
#
## set up the ROI for tracking
#roi = frame[r:r+h, c:c+w]
#hsv_roi =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
#roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
#cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
#
## Setup the termination criteria, either 10 iteration or move by atleast 1 pt
#term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
### end CAMSHIFT
#%% mouse events
drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1

# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
            else:
                cv2.circle(img,(x,y),5,(0,0,255),-1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
        else:
            cv2.circle(img,(x,y),5,(0,0,255),-1)
img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)

while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('m'):
        mode = not mode
    elif k == 27:
        break

cv2.destroyAllWindows()
#%% save an image as template (then crop it)
templatename="/home/m/Dropbox/maze/templates/center1.jpg"
if 0:
    img = read()
    cv2.imwrite(templatename,img)
#%% template matching
from matplotlib import pyplot as plt

img = read()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img.shape
#%
#img=np.asarray(img, dtype=np.float32)
img2 = img.copy()
template = cv2.imread(templatename,0)
template.shape
#template=np.asarray(template, dtype=np.uint8)
w, h = template.shape[::-1]
# All the 6 methods for comparison in a list
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for meth in methods:
    img = img2.copy()
    method = eval(meth)

    # Apply template Matching
    res = cv2.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(img,top_left, bottom_right, 255, 2)

    plt.subplot(121),plt.imshow(res,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)

    plt.show()

#%% WHAT ALREADY WORKS
dobg=1
dodepth=0
if dodepth:
    parnoise=2.7
else:
    parnoise=0
def getbg():        
    global fgbg
    fgbg = cv2.createBackgroundSubtractorMOG(5000,6, .5, parnoise) #int history, int nmixtures, 
    #double backgroundRatio, double noiseSigma=0
if dobg:
    getbg()    
#ret, frame = cap.read()

def read():
    (depth,_), (rgb,_) = get_depth(), get_video()
    if dodepth:
        da=np.dstack((depth,depth,depth)).astype(np.uint8) # this is correct depth 
        frame=np.array(da[::1,::1,::-1]);
    else:
        frame=rgb[::1,::1,::-1]
#    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #frame=np.array() ### !!!! or rgb
    
    
    #fgmask = fgbg.apply(frame)
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',frame)
    k = cv2.waitKey(30) & 0xff
    return frame
    #if k == 27:
    #    break
def skip():
    for i in range(1,100):
        print i
        read()
def readbg():
    #ret, frame = cap.read()
    frame=read();
    if dobg:
        fgmask = fgbg.apply(frame)
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #frame=togray(frame)
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #frame=frame-fgmask
    
    cv2.imshow('frame',gray)
    k = cv2.waitKey(30) & 0xff

#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
def readbgsurf10():
    for i in range(1,20):
        readbgsurf()
def readbgblob10():
    for i in range(1,20):
        x=readbgblob()        
        print x
def readbgblob100():
    for i in range(1,100):
        x=readbgblob()        
        print x
def readbgblobwhile():
    fgbg = getbg()
    while 1:
        x=readbgblob()        
        print x        
#% SIMPLEBLOB
params = cv2.SimpleBlobDetector_Params()
# params.minDistBetweenBlobs = 500.0
params.minArea = 100.0 #10000
params.maxArea = 2000
params.filterByColor = 0
params.filterByArea = 1
params.filterByCircularity = 0
params.filterByConvexity = 0
params.filterByInertia = 0
params.blobColor = 1
params.minThreshold = 0.
params.maxThreshold = 520. #def 220
params.thresholdStep=10#def 10
surf = cv2.SimpleBlobDetector(params)    
def readbgblob():
    global kp
    #ret, frame = cap.read()
    #frame=cv2.resize(frame, (0,0), fx=0.5, fy=0.5) 
    frame=read()
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if dobg:
        fgmask = fgbg.apply(frame)
        img=fgmask      
    else:
        img=frame
    kp = surf.detect(img)
    #kp[1].size
    #kp[0].pt[0]
    # sort here!
    del kp[:-1]
    #print kp[0].pt
    img2=cv2.drawKeypoints(img,kp,None,(255,0,0),4)
    #plt.imshow(img2),plt.show(),plt.draw()
    #plt.imshow(img),plt.show(),plt.draw()
    #cv2.imshow('frame',frame)
    cv2.imshow('bgoff_blobs',img2)
    k = cv2.waitKey(30) & 0xff
    try:
        return kp[0].pt
    except:
        return 0    
    
##%% KALMAN
#from pykalman import KalmanFilter
#import pylab as pl
##### initiate kalman for a few measurements:
##def kalminit():
#kf = KalmanFilter(transition_matrices = [[1, 1], [0, 1]], observation_matrices = [[0.1, 0.5], [-0.3, 0.0]])    
#measurements = np.asarray([[1,0], [0,0], [0,1]])  # 3 observations
#kf = kf.em(measurements, n_iter=5)
#observations=measurements
##### online est
#n_timesteps=3
#for t in range(n_timesteps - 1):
#    if t == 0:
#        filtered_state_means[t] = kf.initial_state_mean
#        filtered_state_covariances[t] = kf.initial_state_covariance
#    filtered_state_means[t + 1], filtered_state_covariances[t + 1] = (
#        kf.filter_update(
#            filtered_state_means[t],
#            filtered_state_covariances[t],
#            observations[t + 1],
#            transition_offset=kf.transition_offsets[t],
#        )
#    )
#
## draw estimates
#pl.figure()
#lines_true = pl.plot(observations[:,0],observations[:,1], '-*',color='b')
#lines_filt = pl.plot(filtered_state_means[:,0],filtered_state_means[:,1], color='r')
#pl.legend((lines_true[0], lines_filt[0]), ('true', 'filtered'))
#pl.show()
#%% bg, blobs, then kalman
# collect some data and init:
#measurements = array(f);
#measurements=np.asarray([[1,0], [0,0], [0,1]],dtype=np.float64)  # 3 observations
from pykalman import KalmanFilter
import pylab as pl
numobs=30
measurements=zeros((numobs,2)) 
#measurement = ones((3,3),Float)
for i in range(0,numobs):
    measurements[i,]=readbgblob()
    print measurements[i,]
#%%    
tr=numpy.array([[1, 1], [0, 0]],'f')
daeye=numpy.array([[1, 1], [1, 1]],'f')
kf = KalmanFilter(
    transition_matrices = 1.3*tr,
    observation_matrices = 2.2*daeye,
    em_vars=[
      'transition_matrices', 
      'observation_matrices',
      'transition_covariance', 
      'observation_covariance',
      'observation_offsets', 
      #'initial_state_mean',
      #'initial_state_covariance'
    ]
)
kf = kf.em(measurements, n_iter=5)    
print kf.transition_matrices
print kf.observation_covariance
print kf.observation_matrices
print kf.transition_covariance
#print kf.observation_offsets
#%% actual online est:
#### online est
n_timesteps=30
measurementsRT=zeros((n_timesteps,2)) 
n_dim_state = 2#kf.transition_matrix.shape[0]
filtered_state_means = np.zeros((n_timesteps, n_dim_state))
filtered_state_covariances = np.zeros((n_timesteps, n_dim_state, n_dim_state))
#filtered_state_means=measurementsRT
#filtered_state_covariances=measurementsRT
for t in range(n_timesteps - 1):
    measurement=readbgblob()
    measurementsRT[t,:]=measurement
    if t == 0:
        filtered_state_means[t,] = measurements[i,]#kf.initial_state_mean
        filtered_state_covariances[t,] = [[70,70],[70,70]]#kf.initial_state_covariance
    filtered_state_means[t + 1,], filtered_state_covariances[t + 1,] = (
        kf.filter_update(
            filtered_state_means[t,],
            filtered_state_covariances[t,],
            measurementsRT[t,]#,
            #transition_offset=kf.transition_offsets[0], # originally data.transition_offsets[t]
            #observation_offset=kf.observation_offsets
        )
    )
    print filtered_state_means[t+1]
    print measurementsRT[t]

#%% draw estimates
pl.figure()
lines_true = pl.plot(measurementsRT[:,0],measurementsRT[:,1], '-*',color='b')
lines_filt = pl.plot(filtered_state_means[:,0],filtered_state_means[:,1], color='r')
pl.legend((lines_true[0], lines_filt[0]), ('measured', 'filtered'))
pl.show()
#%% working: zmq sockets for OE
import zmq
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://127.0.0.1:5556")
#socket.send('mmyros')
#%% 
socket.send('TrialStart')
#%%
socket.send('TrialEnd')
#%%
socket.send('StartRecord')
#%% 
socket.send('StopRecord')
#%% sockets for OE
import zmq
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://127.0.0.1:5000")
 
#while True:
 msg = socket.recv()
 print "Got", msg
 socket.send(msg)
#%% SERIAL FOR OE
import os, pty, serial
master, slave = pty.openpty()
s_name = os.ttyname(slave)
ser = serial.Serial(s_name)

#%% To Write to the device
ser.write('EYE_POSITION 10,23,34')
#%% To read from the device
os.read(master,1000)
#%% stock serial, no work
import serial
ser = serial.Serial('/dev/ttyS1', 19200, timeout=1)  # open first serial port
print ser.name          # check which port was really used
ser.write("hello")      # write a string
ser.close()             # close port
#%%
##%% EDGES v2    - no work either
#def readbgedges():
#    ret, frame = cap.read()
#    frame=cv2.resize(frame, (0,0), fx=0.5, fy=0.5) 
#    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#    foremat = fgbg.apply(frame)    
#    cv2.imshow('frame',foremat)
#    k = cv2.waitKey(30) & 0xff
#    #kp[1].size
#    #kp[0].pt[0]
#    # sort here!
#    #print kp[0].pt
#    #img2=cv2.drawKeypoints(frame,kp,None,(255,0,0),4)
#    #plt.imshow(img2),plt.show(),plt.draw()
#    #plt.imshow(img),plt.show(),plt.draw()
#    #cv2.imshow('frame',frame)
#    #cv2.imshow('frame',img2)
#    #k = cv2.waitKey(30) & 0xff    
#    ret,thresh = cv2.threshold(foremat,127,255,0)
#    contours = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#    if len(contours) > 0:
#        m= np.mean(contours[0],axis=0)
#        measuredTrack[count-1,:]=m[0]
#        #plt.plot(m[0,0],m[0,1],'ob')
#    
#    
##%% EDGGES NO WORK
#numframes=10
#measuredTrack=np.zeros((numframes,2))-1
#count=0
#history = 10
#nGauss = 3
#bgThresh = 0.6
#noise = 20
#while count<numframes:
#    ret, frame = cap.read()
#    bgs = cv2.createBackgroundSubtractorMOG(history,nGauss,bgThresh,noise)
#    count+=1
#    img2 = frame
#    #img2 = capture.read()[1]
#    cv2.imshow("frame",img2)
#    foremat=bgs.apply(img2)
#    cv2.waitKey(100)
#    imgray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#    ret,thresh = cv2.threshold(imgray,127,255,0)
#    contours= cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#    foremat=bgs.apply(img2)
#    #ret,thresh = cv2.threshold(img2,127,255,0)
#    #contours = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#    cv2.drawContours(foremat,contours,-1,(0,255,0),3)
#    if len(contours) > 0:
#        m= np.mean(contours[0],axis=0)
#        measuredTrack[count-1,:]=m[0]
#        #plt.plot(m[:0],m[:1])
#        cv2.imshow('frame',foremat)
#        cv2.waitKey(80)
#    
##%% SURF 2 - TOO SLOW
#surf = cv2.SURF(400)    
#def readbgsurf():
#    ret, frame = cap.read()
#    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#    fgmask = fgbg.apply(frame)
#    img=fgmask
#    surf = cv2.SURF(400)    
#    #kp, des = surf.detectAndCompute(img,None)
#    #len(kp)
#    #print surf.hessianThreshold
#    # We set it to some 50000. Remember, it is just for representing in picture.
#    # In actual cases, it is better to have a value 300-500
#    surf.hessianThreshold = 5000
#    # Again compute keypoints and check its number.
#    kp, des = surf.detectAndCompute(img,None)
#    print len(kp)
#    img2=cv2.drawKeypoints(frame,kp,None,(255,0,0),4)
#    #plt.imshow(img2),plt.show(),plt.draw()
#    #cv2.imshow('frame',gray)
#    cv2.imshow('img2',img2)
#    k = cv2.waitKey(30) & 0xff
##%% camshift# take first frame of the video
#ret,frame = cap.read()
#
## setup initial location of window
#r,h,c,w = 250,90,400,125  # simply hardcoded the values
#track_window = (c,r,w,h)
#
## set up the ROI for tracking
#roi = frame[r:r+h, c:c+w]
#hsv_roi =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#mask = cv2.inRange(hsv_roi, np.array((0., 6.,3.)), np.array((180.,255.,255.)))
#roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
#cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
#
## Setup the termination criteria, either 10 iteration or move by atleast 1 pt
#term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
#
#while(1):
#    ret ,frame = cap.read()
#
#    if ret == True:
#        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
#
#        # apply meanshift to get the new location
#        ret, track_window = cv2.CamShift(dst, track_window, term_crit)
#
#        # Draw it on image
#        pts = cv2.boxPoints(ret)
#        pts = np.int0(pts)
#        img2 = cv2.polylines(frame,[pts],True, 255,2)
#        cv2.imshow('img2',img2)
#
#        k = cv2.waitKey(60) & 0xff
#        if k == 27:
#            break
#        else:
#            cv2.imwrite(chr(k)+".jpg",img2)
#
#    else:
#        break
#
#cv2.destroyAllWindows()
#cap.release()


##%% SURF
#img=frame    
#surf = cv2.SURF(400)    
#kp, des = surf.detectAndCompute(img,None)
#len(kp)
#print surf.hessianThreshold
#400.0
#
## We set it to some 50000. Remember, it is just for representing in picture.
## In actual cases, it is better to have a value 300-500
#surf.hessianThreshold = 10000
#
## Again compute keypoints and check its number.
#kp, des = surf.detectAndCompute(img,None)
#
#print len(kp)
#img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)
#plt.imshow(img2),#plt.show()
#cv2.imshow('img2',gray)
#
##%%    
#def readcamsh():    
#    #%
#    global track_window
#    ret, frame = cap.read()
#   
#    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#    dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
#
#    # apply meanshift to get the new location
#    ret, track_window = cv2.CamShift(dst, track_window, term_crit)
#    x,y,w,h = track_window
#    cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
#    cv2.imshow('frame',frame)
#
#    k = cv2.waitKey(60) & 0xff
##%%    
#def readcamsh():    
#    #%
#    global track_window
#    #ret, frame = cap.read()
#   
#    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#    dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
#
#    # apply meanshift to get the new location
#    ret, track_window = cv2.CamShift(dst, track_window, term_crit)
#    x,y,w,h = track_window
#    cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
#    cv2.imshow('frame',frame)
#
#    k = cv2.waitKey(60) & 0xff
#
##    # Draw it on image
##    pts = cv2.boxPoints(ret)
##    pts = np.int0(pts)
##    cv2.polylines(frame,[pts],True, 255,2)
##    cv2.imshow('frame',frame)
##
##    k = cv2.waitKey(60) & 0xff
#    
##%% meanshift    
## take first frame of the video
#ret,frame = cap.read()
#
## setup initial location of window
#r,h,c,w = 30,30,400,125  # simply hardcoded the values
#track_window = (c,r,w,h)
#
## set up the ROI for tracking
#roi = frame[r:r+h, c:c+w]
#hsv_roi =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
#roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
#cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
#
## Setup the termination criteria, either 10 iteration or move by atleast 1 pt
#term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
#
#ret ,frame = cap.read()
#
#if ret == True:
#    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#    dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
#
#    # apply meanshift to get the new location
#    ret, track_window = cv2.meanShift(dst, track_window, term_crit)
#
#    # Draw it on image
#    x,y,w,h = track_window
#    cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
#    #print(cv2.rectangle(frame, (x,y), (x+10,y+10),(0,0,255)))
#    cv2.imshow('frame',frame)
#
#    k = cv2.waitKey(60) & 0xff    
#    
## while(1):
##    ret, frame = cap.read()
#
#    # fgmask = fgbg.apply(frame)
#
#    # cv2.imshow('frame',fgmask)
#    # k = cv2.waitKey(30) & 0xff
#    # if k == 27:
#    #     break
##%%
#cap.release()
#cv2.destroyAllWindows()
