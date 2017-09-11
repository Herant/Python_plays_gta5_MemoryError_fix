# create_training_data.py

import numpy as np
from grabscreen import grab_screen
import cv2
import time
from getkeys import key_check
import os
import gc



def keys_to_output(keys):
    '''
    Convert keys to a ...multi-hot... array
    [A,W,D] boolean values.
    '''
    output = [0,0,0]
    
    if 'A' in keys:
        output[0] = 1
    elif 'D' in keys:
        output[2] = 1
    else:
        output[1] = 1
    return output


file_name = 'training_data.npy'


if os.path.isfile(file_name):
    print('File exists, loading previous data!')
    training_data = list(np.load(file_name))
else:
    print('File does not exist, starting fresh!')
    training_data = []


def main():

    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)


    paused = False
    while(True):

        if not paused:
            # 800x600 windowed mode
            screen = grab_screen(region=(0,40,800,640))
            last_time = time.time()
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (40,40))
            # resize to something a bit more acceptable for a CNN
            keys = key_check()
            output = keys_to_output(keys)
            training_data.append([screen,output])
            
            if len(training_data) & 1000 == 0:
                print(len(training_data))
                np.save(file_name,training_data)
                gc.enable()
                gc.collect()
                if len(training_data) >= 10000:     #CHANGE THIS TO 20000, 30000 ..etc after each session
                    break
                    gc.enable()
                    gc.collect()
            
        keys = key_check()
        if 'T' in keys:
            if paused:
                paused = False
                print('unpaused!')
                time.sleep(1)
            else:
                print('Pausing!')
                paused = True
                time.sleep(1)
        elif 'Q' in keys:
            break


main()


'''
**********************  READ THIS  ***********************

The first thing is to import gc, which is built in garbage collector, i'm not sure if gc helps at all but i like to think that it does.

now change following: 

screen = cv2.resize(screen, (40,40))

after that add few lines:

if len(training_data) % 1000 == 0:
                print(len(training_data))
                np.save(file_name,training_data)
                gc.enable()
                gc.collect()
                if len(training_data) >= 10000:
                    break

i'd strongly suggest to make a copy of .npy file after each session, and if it crashed and delete your original "training_data.npy" you'd still have the copy from last session which you can continue from.
I increased every session with  10000, and with this setup i got to 79000 which was alot better than getting max 8000..

also i did threw:

gc.enable()
gc.collect()

in my shell after each session, like i said i'm not sure if it helps but whatever..

If there is anyone who got better solution please be my guest.'''
