 # -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 00:16:28 2014
Definitions for RAM RT-video processing 
@author: m
"""

#%% load arduino definitions from a def file
#runfile('/home/m/Dropbox/maze/ramarpydefs.py')
#%% imports, capture
dokinect=0
import numpy as np
import random
import time
import cv2
import matplotlib.pyplot as plt
import matplotlib
import pylab
import csv
import httplib, urllib
import pushover
import thread
import scipy.io as sio

def send_note(msg):
	pushover.init("acfZ42h7KMGmAdbzyCBZxkDwTrzhPN")
	client = pushover.Client("uxFdSnAMc9D9kcBdgZWYkW3mwynUvc")
	client.send_message(msg, title=msg, priority=1)
def send_note_old(msg):
	#msg= "hello world"
	conn = httplib.HTTPSConnection("api.pushover.net:443")
	conn.request("POST", "/1/messages.json",
  	urllib.urlencode({
   	 "token": "acfZ42h7KMGmAdbzyCBZxkDwTrzhPN",
   	 "user": "uxFdSnAMc9D9kcBdgZWYkW3mwynUvc",
   	 "message": msg,
  	}), { "Content-type": "application/x-www-form-urlencoded" })
	conn.getresponse()
#%%
# Define the codec and create VideoWriter object
dafilename='/home/m/temp/temp'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(dafilename+'.avi',fourcc, 10.0, (640,480)) 
#%%
if dokinect:
    from freenect import sync_get_depth as get_depth, sync_get_video as get_video
else:
    cap = cv2.VideoCapture(1)
##### RECORDED VIDEO: ######
#cap = cv2.VideoCapture('/home/m/Dropbox/maze/video/examples/3.mp4')
##### CAPTURE    ########
#cap = cv2.VideoCapture(1)
#cap = cv2.cvCaptureFromCAM()
#fgbg = cv2.createBackgroundSubtractorMOG2()
#%%
dobg=1
dodepth=0

if dodepth:
    parnoise=2.7
else:
    if dokinect:
        parnoise=0
        parbgratio=.5;
    else:
        parnoise=1
        parbgratio=.05; # smaller for longer tracking of static object
def getbg():        
    global fgbg
    fgbg = cv2.createBackgroundSubtractorMOG(500,6, parbgratio, parnoise) #int history, int nmixtures, 
    return fgbg
    #double backgroundRatio, double noiseSigma=0
if dobg:
    getbg()    
def readweb():
    ret, frame = cap.read()
    #fgmask = fgbg.apply(frame)
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',frame)
    k = cv2.waitKey(10) & 0xff        
    return frame
def readkinect():
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
#ret, frame = cap.read()
def read():
    global frame
    if dokinect:
        frame=readkinect()
    else:
        frame=readweb()
    out.write(frame)        
    return frame
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
    
    cv2.imshow('frame',frame)
    k = cv2.waitKey(10) & 0xff

    
#% SIMPLEBLOB
params = cv2.SimpleBlobDetector_Params()
# params.minDistBetweenBlobs = 500.0
params.minArea = 200.0 #10000
params.maxArea = 1400
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
    #    try:
    #        print kp[0].size
    #    except:
    #        pass            
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
    if not kp:
        print 'no blobs'
        return (0,0) 
    try:
#        print kp[0].pt
        return kp[0].pt
    except:        
        print 'error in readbgblob'
        return (0,0)   

#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
#def readbgsurf10():
#    for i in range(1,20):
#        readbgsurf()
def readbgblob10():
    for i in range(1,20):
        x=readbgblob()        
        print x
def readbgblob100():
    for i in range(1,100):
        x=readbgblob()        
        print x
def readbgbwhile():
    fgbg = getbg()
    while 1:
        x=readbgblob()        
        print x        
#        if isinstance(x, tuple):
#            if x[1]<400:
#                moveall(4,1)
#            else:
#                moveall(4,0)
    return x
# match template:
    
def center_from_rectangle(top_left,w,h,img):
    center = (top_left[0] + w/2, top_left[1] + h/2)
    cv2.circle(img,center,w/2,0)
    plt.imshow(img,cmap = 'gray')
    plt.show()
    return center
def get_match(templatename):
    for i in range(1,7):
        img=read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(templatename,0)
    w, h = template.shape[::-1]
    meth='cv2.TM_CCOEFF_NORMED'
    method=eval(meth)
    res = cv2.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    center=center_from_rectangle(top_left,w,h,img);
    return center    
def get_center():
    templatename="/home/m/Dropbox/maze/templates/center1.jpg"
    center=get_match(templatename)
    return center

#%% (x1-x2)^2-(y1-y2)^2 implementation:
def dista(i1,i2):
    x1=i1[0]
    x2=i2[0]
    y1=i1[1]
    y2=i2[1]
    o=(x1-x2)**2+(y1-y2)**2
    return o

# close door if in arm
def close_when_in():
    global center 
    x=0;
    while dista(x,center)<100:
        x=readbgblob()
#%% match to arms:
def get_arm_coord(narm):
    templatename="/home/m/Dropbox/maze/templates/arm"+str(narm)+".jpg"        
    o=get_match(templatename)
    print o
    return o        
def get_arm_coords():
    armcoords=[]
    for narm in range (1,9):
        armcoords.append(get_arm_coord(narm))
    return armcoords
#%% look up which arm the rat is in if precisely in center of arm:
def which_arm(curcoord,armcoords):
    narmout=armcoords.index(curcoord)+1 # +1 bec python starts indices at 0
#    narmout=(armcoords==curcoord).nonzero()
    print narmout
    return narmout
#%% look up where he is with leeway: practical implementation
def which_arm_near(curcoord,armcoords):
    global center
    try:
        fromcenter=dista(curcoord,center)
    except:
        narmout=999
        print 'unknown => 999'
        return  narmout
    if fromcenter<3000: #then still in center, did not commit
        print 'center => 0'
        narmout=0
        return narmout
    else:
        #print 'not center'
        for narm in range(1,9):
            curdist=dista(armcoords[narm-1],curcoord)
            if curdist<4000:
                narmout=narm
                print narmout
                return narmout
        print 'uncommitted => 111'
        narmout=111
        return narmout        
def iserror(prevnarm,curnarm,targets):
    if prevnarm!=curnarm and curnarm!=111 and curnarm!=0 and curnarm!=999:
        if curnarm in targets:
            return 0    
        else:   
            return 1        
    else:
        return 0
    
#%%        
def is_near_feeder(curcoord,curnarm):        
    global center
    try:
        dadist=dista(curcoord,center)
        print dadist
#        if curnarm==1 or curnarm == 2:
#            distcutoff = 15000 
#        elif curnarm==5 or curnarm==6:
#            distcutoff = 20000 
#        else:
	distcutoff = 24000         
        if dadist>distcutoff and curcoord!=0: 
            is_near=1
        else:
            is_near=0
    except:
        is_near=0
    print 'is_near:'
    print is_near            
    return is_near
#%% arm coordinates:
#armcoords=get_arm_coords()                
#%% get bg and go            
armcoords=get_arm_coords()   
center=get_center()
def readbgblobcoord():
    getbg()
    while True:
        curcoord=readbgblob()
        curnarm=which_arm_near(curcoord,armcoords)    
        return curnarm
#%% no taking background:                
def readblobcoord():
    while True:
        curcoord=readbgblob()
        curnarm=which_arm_near(curcoord,armcoords)   
d=[]        
keys = ['trials', 
	'xcoord',
	'ycoord', 
	'state',
	'arms_visited', 
	'curnarm', 
	'doors_to_open',
	'ts','center',
	'armcoords',
	'didtimeout',
	'diderrors',
	'targets']
d.append({"center":center,
          "armcoords" :armcoords
          })
def write_data():
    global d
    d.append({"trials":      trials,
             "xcoord" :       curcoord[0],
             "ycoord" :       curcoord[1],
             "state" :       state,
             "arms_visited": arms_visited,
             "curnarm":      curnarm,
             "doors_to_open":doors_to_open,
             "ts":           time.time(),
             "didtimeout":   didtimeout,
             "diderrors":    diderrors,
             "targets":      np.array(list(targets))
             })            

def input_thread(L):
    raw_input()
    L.append(None)
#%%         
#%% state machine programs:
def ram_6_2(ntrials,dotakebg):
    global state
    global trials
    global curcoord, arms_visited,doors_to_open,curnarm
    global out,didtimeout,diderrors,targets
    closeall1()
    def dowait():
        L = []
        thread.start_new_thread(input_thread, (L,))
        print 'waiting'
        while 1:
            curcoord=readbgblob()
            curnarm=which_arm_near(curcoord,armcoords)
            write_data()
            if L: break
        print '----------'  
        print 'gotcha'
        # another 10 secs of quiet
        now = time.time()
        while_starts = now
        while (now-while_starts<10):
            now = time.time()
            curcoord=readbgblob()
            curnarm=which_arm_near(curcoord,armcoords)
            write_data()
        print 'done waiting'
    if dotakebg:
	    for i in range(1,15):
		    getbg()
    closeall1()
    trials=0;
    curcoord=(0,0);
    state="0"
    arms_visited=0
    curnarm=0
    doors_to_open=0
    didtimeout=0
    diderrors=0
    targets=(0,0)
    # wait for the rat to get in 
    dowait()
    print '--------------------'
    print 'rat is in, lets go!'
    for trials in range(0,ntrials):
        datime=time.strftime("%d-%m-%Y--%H:%M:%S")
        daname='ramtest_'+datime
        dapath='/home/m/data/ramp/'
        dafilename=dapath+daname 
        print dafilename
        # Define the codec and create VideoWriter object
        out.release()
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(dafilename+'.avi',fourcc, 10.0, (640,480)) 
        print 'out ok'
        didtimeout=0
        diderrors =0
        trial_starts = time.time()
        now = time.time()
        closeall1()
        #% WAIT to get to center
        state="wait"
        #% generate 4 rand integers from 1 to 8     and close those doors    
        doors_to_open=random.sample(range(1,9), 6)#[randint(1,8) for p in range(0,4)]
        targets=doors_to_open
        print doors_to_open        
        arms_to_visit=set(range(1,9))-set(doors_to_open)
        print arms_to_visit
       
        while state=="wait":

            print state,trials
            curcoord=readbgblob()
            print curcoord
            curnarm=which_arm_near(curcoord,armcoords)
            if curnarm==0:
                state='1st6'
                print '========================'
                print state
                print '========================'
            write_data()
        #% 1st4: open four random DOORS        
        for door in doors_to_open:
            moveall(door,1) 
        arms_visited=[];    
        now = time.time()       
        elapsed=now-trial_starts
        while state=='1st6':# and elapsed<60*5: # the trial has to be less than 5 mins:
            now = time.time()       
            elapsed=now-trial_starts          
            if elapsed>60*5:
                timeout=1
            print elapsed/60,state,trials,arms_visited,'of',set(range(1,9))-set(arms_visited)
            prevnarm=curnarm
            curcoord=readbgblob()
            curnarm=which_arm_near(curcoord,armcoords)
            diderror=iserror(prevnarm,curnarm,targets)
            if (curnarm in arms_visited)==False and (curnarm!=999) and (curnarm!=0)  and (curnarm!=111)  :
                arms_visited.append(curnarm)
            if len(arms_visited)>5 and is_near_feeder(curcoord,curnarm):
                state='delay'                
                print '========================'
                print state,trials,arms_visited
            write_data()
	#% delay
        closeall1()
        while_starts = time.time()
        now = time.time()
        while(now-while_starts<60):# and elapsed<60*5: # the trial has to be less than 5 mins:
            now = time.time()       
            elapsed=now-trial_starts;
            now = time.time()
            print elapsed/60
            print("Waiting for {0} seconds out of 60".format(now - while_starts)) 
            curcoord=readbgblob()
            curnarm=which_arm_near(curcoord,armcoords)
            write_data()
	#%% second set
        state='2nd2'
        arms_visited=list()
        arms_to_visit=set(range(1,9))-set(doors_to_open)
        targets=arms_to_visit
        print arms_to_visit
        openall()
        send_note(state)
        now = time.time()       
        elapsed=now-trial_starts
        while state=='2nd2':# and elapsed<60*5: # the trial has to be less than 5 mins:
            #if elapsed<60*5:
            #    timeout=1
            now = time.time()       
            elapsed=now-trial_starts
            print elapsed/60, state,trials,arms_to_visit, ' of ',arms_visited
            prevnarm=curnarm
            curcoord=readbgblob()
            curnarm=which_arm_near(curcoord,armcoords)
            diderror=iserror(prevnarm,curnarm,targets)
            write_data()
            if (curnarm in arms_visited)==False and (curnarm in arms_to_visit) and (curnarm!=999) and (curnarm!=0)  and (curnarm!=111)  :
                arms_visited.append(curnarm)
                send_note("visited arm")
            if len(arms_visited)>1 and curnarm==0:
                state='wait'
                closeall1()
                break
        closeall1()
        send_note("intertrial")   
        print 'all done, next trial please'
        #% save data to disk
        sio.savemat('/home/m/Dropbox/maze/'+daname,{'d':d})
        f = open('/home/m/Dropbox/maze/'+daname+'.csv', 'wb')    
        dict_writer = csv.DictWriter(f,keys)
        dict_writer.writer.writerow(keys)
        dict_writer.writerows(d)
	
        print 'saved'
        #% between trial wait  
        print 'intertrial period. Come feed me!'
        dowait()
    out.release()
