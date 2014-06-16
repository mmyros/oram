# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 13:48:41 2014

@author: m
"""
# 4 noisy but works
# 
#%% load arduino definitions from a def file
#execfile("ramarpydefs.py")
#% load opencv definitions
execfile('ramviddefs_notpushy.py')
#% move servo's pin 4 to closed position
#moveall(4,0)
#% start the background job
startthread()    
#%% do something while the thing is running
#blinkfast13()
#%% stop workers
#stopthreads()
#%% run gui
#argui()
#%% sometimes you have to restart the board
#board = Arduino('/dev/ttyACM1')
