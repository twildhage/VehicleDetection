#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 20 02:43:19 2017

@author: timo
"""
from enum import Enum
from collections import deque
from itertools import count
import numpy as np


class VehicleDetector(object):
    def __init__(self):
        self.cars = []
        self.feature_vectors = np.array()
        self.feature_distances = []
        self.spatial_distances = []
        
        
    # full scan:
        # perform full image scan
        # sum matched windows weighted by prediction
        # threshold max layers
        # extract bounding window
        # create new Cars(window) 
    # local scan:
        # create local scan windows
        # perform local image scans
        # extract features
        # store feature matrix
        # store spatial matrix
        # calculate feature distances
        # calculate spatial distances
        # compare features and radius_inner_circle
    
    # if True: 
        # filter features
        # append features
        # append images
        # append windows
        # delete matched window
        # draw colored window
        # draw annotation
        # draw colored overlay
        # update inner and outer circle radiuses
        # reset car_lost_counter

    # if False:
        # predict window position
        # draw colored window
        # draw annotation
        # no overlay
        # increase car_lost_counter
        
        # if car_lost_counter > max:
            # delete car object
            # switch to full scan
            
    # if unattachted matches left:
        # if distance to all cars greater than radius_outer_circle
        # create new car objects
    # increase decrease timesteps_until_full_scan

    # if timesteps_until_full_scan > timesteps_max:
        # switch to full scan
        # reset timesteps_until_full_scan
    
class Cars(object):
    
    car_ids = count(0)
    
    def __init__(self, image, window):
        self.feature_vector = deque(maxlen=2)
        self.image    = deque(maxlen=2)
        self.prediction = deque(maxlen=2)
        self.window = deque(maxlen=2)
        self.color  = Color.GREEN
        self.annotation = {'testfield': '', 'coordinates': []}
        self.id = next(self.car_ids)
        self.radius_inner_circle = 20
        self.radius_outer_circle = 40
        
        
        
class Color(Enum):
    RED = 1
    GREEN = 2