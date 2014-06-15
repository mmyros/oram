# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 18:16:02 2014

@author: root
"""

#%% init 
from pyfirmata import Arduino, util
from Tkinter import *
import time as t
board = Arduino('/dev/ttyACM1')

global commands
#%% function that controls arduino
def a (pin , value):
    """ pin , value
    """
    board.digital[pin].write(value)
    #return 0

#%% define the blink function    
def blink():
    a(4, 0);
    t.sleep(1)
    a(4, 1);
    t.sleep(1)
def blinkfast():
    for x in range(0, 12):
        a(4, 0);
        t.sleep(.1)
        a(4, 1);
        t.sleep(.1)
def blinkfast13():
    for x in range(0, 12):
        a(13, 0);
        t.sleep(.1)
        a(13, 1);
        t.sleep(.1)
#% main function that will run in the thread:    
def mainfun():
    argui()
#    ram_6_2(15)    
#    maze1();
#%% control a servo
#%% set up pinlist as a bunch of servo objects
pinlist=[];
for i in range(2,10)    :
    #%
    pin = board.get_pin('d:' + `i` + ':s')
    pinlist.append(pin)
    pin.write(43)
#%% move pins by list
def moveall(dapin,a):
    # 28 degrees to close, 92 to open
    if a == 1:
        degr=85;
    elif dapin==2 or dapin==7:
        degr=43;
    else:
        degr=43;    
    pinlist[dapin-1].write(degr)    # -2 fixes the factthat we start with pin 2, and python counts arrays starting with 0
def movealldegr(dapin,degr):
    # 28 degrees to close, 92 to open
    pinlist[dapin-1].write(degr)    # -2 fixes the factthat we start with pin 2, and python counts arrays starting with 0
def closeall():
    for i in range(1,9):
        moveall(i,0)
def closeall1():        
    movealldegr(1,46)
    movealldegr(2,43)
    movealldegr(3,32)
    movealldegr(4,40)
    movealldegr(5,37)
    movealldegr(6,24)
    movealldegr(7,39)
    movealldegr(8,33)
def openall():
    for i in range(1,9):
        moveall(i,1)
def maze1():
    for i in 1,3,5,7:
        moveall(i,1)
    t.sleep(5)
    closeall()
    t.sleep(5)
    for i in 2,4,6,8:
        moveall(i,0)
    t.sleep(10)
    
#moveall(2,0)    
##%% move pin 4
#pin4 = board.get_pin('d:4:s')
#def move4(a):
#    #
#    # 28 degrees to close, 92 to open
#    if a == 1:
#        degr=85;
#    else:
#        degr=33;
#    pin4.write(degr)
#% functions for the sliders
def m1(degr):
    movealldegr(1,degr) 
def m2(degr):
    movealldegr(2,degr) 
def m3(degr):
    movealldegr(3,degr) 
def m4(degr):
    movealldegr(4,degr) 
def m5(degr):
    movealldegr(5,degr) 
def m6(degr):
    movealldegr(6,degr) 
def m7(degr):
    movealldegr(7,degr) 
def m8(degr):
    movealldegr(8,degr) 

#%% set up GUI
def argui():
    root=[];
    root = Tk()
     
    #% draw a nice big slider for servo position
    scale1 = Scale(root,
        command = m1,
        to = 175,
        orient = HORIZONTAL,
        length = 400,
        label = 'Arm1')
    scale1.set(46)
    scale1.pack(anchor = CENTER)
    scale1 = Scale(root,
        command = m2,
        to = 175,
        orient = HORIZONTAL,
        length = 400,
        label = 'Arm2')
    scale1.set(43)
    scale1.pack(anchor = CENTER)
    scale1 = Scale(root,
        command = m3,
        to = 175,
        orient = HORIZONTAL,
        length = 400,
        label = 'Arm3')
    scale1.set(32)
    scale1.pack(anchor = CENTER)
    scale1 = Scale(root,
        command = m4,
        to = 175,
        orient = HORIZONTAL,
        length = 400,
        label = 'Arm4')
    scale1.set(40)
    scale1.pack(anchor = CENTER)
    scale1 = Scale(root,
        command = m5,
        to = 175,
        orient = HORIZONTAL,
        length = 400,
        label = 'Arm5')
    scale1.set(37)
    scale1.pack(anchor = CENTER)
    scale1 = Scale(root,
        command = m6,
        to = 175,
        orient = HORIZONTAL,
        length = 400,
        label = 'Arm6')
    scale1.set(24)
    scale1.pack(anchor = CENTER)
    scale1 = Scale(root,
        command = m7,
        to = 175,
        orient = HORIZONTAL,
        length = 400,
        label = 'Arm7')
    scale1.set(39)
    scale1.pack(anchor = CENTER)
    scale1 = Scale(root,
        command = m8,
        to = 175,
        orient = HORIZONTAL,
        length = 400,
        label = 'Arm8')
    scale1.set(33)
    scale1.pack(anchor = CENTER)    
    
    def quit(root):
        root.destroy()
    Button(root, text="Quit", command=lambda root=root:quit(root)).pack()
    root.mainloop()
#Button(root, text="Quit", command=quit).pack()    
# run Tk event loop
#root.mainloop()
#print scale1.get()
    
#%% start a background process with the main function

import threading, time, Queue, termios,sys,tty    
def startthread():
    global commands
    # setting a cross platform getch like function
    def getch() :
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try :
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally :
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch
    
    # this will allow us to communicate between the two threads
    # Queue is a FIFO list, the param is the size limit, 0 for infinite
    commands = Queue.Queue(0)
    
    # the thread reading the command from the user input     
    def control(commands) :
    
        while 1 :
    
            command = getch()
            commands.put(command) # put the command in the queue so the other thread can read it
    
            #  don't forget to quit here as well, or you will have memory leaks
            if command == "q" :
                break
    
    
    # your function displaying the words in an infinite loop
    def display(commands):
    
        #string = "the fox jumped over the lazy dog"
        #words = string.split(" ")
        pause = False 
        command = ""
    
        # we create an infinite generator from you list
        # much better than using indices
        #word_list = itertools.cycle(words) 
    
        # BTW, in Python itertools is your best friend
    
        while 1 :
    
            # parsing the command queue
            try:
               # false means "do not block the thread if the queue is empty"
               # a second parameter can set a millisecond time out
               command = commands.get(False) 
            except Queue.Empty:
               command = ""
    
            # behave according to the command
            if command == "q" :
                break
    
            if command == "p" :
                pause = True
    
            if command == "r" :
                pause = False
    
            # if pause is set to off, then print the word
            # your initial code, rewritten with a generator
            if not pause :
                #os.system("clear")
                #blink()
                mainfun()
                #print word_list.next() # getting the next item from the infinite generator 
    
            # wait anyway for a second, you can tweak that
            #time.sleep(1)
    
    
    
    # then start the two threads
    displayer = threading.Thread(None, # always to None since the ThreadGroup class is not implemented yet
                                display, # the function the thread will run
                                None, # doo, don't remember and too lazy to look in the doc
                                (commands,), # *args to pass to the function
                                 {}) # **kwargs to pass to the function
    
    controler = threading.Thread(None, control, None, (commands,), {})
    #% start the threads
    displayer.start()
    controler.start()    
def stopthreads():
    global commands
    commands.put('q')
closeall()    