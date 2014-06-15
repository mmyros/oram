
#%% WHAT ALREADY WORKS
runfile('/home/m/Dropbox/maze/ramviddefs.py')
#%% get center:
center=get_center()
#%% get arm coordinates:
armcoords=get_arm_coords()
#%% look up which arm the rat is in if precisely in center of arm:
curcoord=armcoords[7] # remember to add 1 for correct answer
narm=which_arm(curcoord,armcoords)
#%% look up where he is with leeway
curnarm=which_arm_near(curcoord,armcoords)
#%% in loop:
readbgblobcoord()
#%% state machine:
ram_6_2(15) # 15 trials