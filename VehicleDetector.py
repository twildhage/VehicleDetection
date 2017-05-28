#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 20 02:43:19 2017

@author: timo
"""
from collections import deque
from itertools import count
import numpy as np
import pipeline as pl
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.ndimage.measurements import label
import cv2


class Detector(object):
      
    def __init__(self, model):
        self.detections = []
        self.model = model
        
    @property
    def local_scan_areas(self):
        areas = []
        for detection in self.detections:
            areas.append(detection.local_scan_windows)
        return areas
        
    @property
    def feature_matrix(self):
        D = []
        for detection in self.detections:
            D.append(detection.feature_vector)
        return np.array(D).squeeze()
    
    @property
    def spatial_matrix(self):
        D = []
        for detection in self.detections:
            xc, yc = detection.center_coords
            D.append((xc,yc))
        return np.array(D)
    
    
    def full_scan(self, image, windows, max_layers, verbose=0):
        hot_windows, predictions, textfields = pl.search_windows(image, windows, self.model, 0.99, verbose=verbose)
        heat = np.zeros_like(image[:,:,0]).astype(np.float32)
        heat = pl.add_heat(heat, hot_windows) 
    
        heat = pl.apply_threshold(heat, max_layers)
        heatmap = np.clip(heat, 0, 255)
        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        boxes = pl.extract_labeled_bboxes(labels)
        
        if verbose==1:
            pass
        return (boxes, predictions)
        
        
    def local_scan(self, image, max_local_layers, verbose=0):
        boxes = []
        predictions = []
        
        for detection in self.detections:
            area = detection.local_scan_windows
            hot_windows, prediction, textfields = pl.search_windows(image, area, self.model, 0.9, verbose=verbose)
            heat = np.zeros_like(image[:,:,0]).astype(np.float32)
            heat = pl.add_heat(heat, hot_windows) 
        
            heat = pl.apply_threshold(heat, max_local_layers)
            heatmap = np.clip(heat, 0, 255)
            # Find final boxes from heatmap using label function
            labels = label(heatmap)
            tmpboxes = pl.extract_labeled_bboxes(labels)
            if len(tmpboxes) > 0:
                detection.car_lost_counter = 0
                predictions.append(prediction)
                detection.update(image, tmpboxes[0], np.mean(prediction))
                boxes.append(detection.window[-1])
                if verbose==1:
                    print("Boxes: {0}, hot_windows: {1}, pred: {2}, heat: {3}, heatmap: {4}".format(len(boxes), len(hot_windows),
                          prediction, heat.shape, heatmap.shape))
            else:
                detection.car_lost_counter +=1
            
        return (boxes, predictions)
    
    
class Detection(object):
    _scalers = {'spatial': StandardScaler(),
               'hist': StandardScaler(),
               'hog': StandardScaler() }
    
    _kw_spatial = {'size':(32, 32)}
    _kw_hist    = {'nbins':16}
    _kw_hog     = {'orient':8, 
              'pix_per_cell':8,
              'cell_per_block':2}
    _kw_extract = {'kw_spatial':_kw_spatial,
                  'kw_hist':_kw_hist,
                  'kw_hog':_kw_hog,
                  'color_space':'HLS',
                  'hog_channel':'ALL',
                  'spatial_feat':False,
                  'hist_feat':True,
                  'hog_feat':True}
    _overlap = 0.2
    _scaling_factor = 2
    
    def __init__(self, image, window, prediction, scalers):
        history_length = 4
        self.image    = deque(maxlen=history_length)
        self.prediction = deque(maxlen=history_length)
        self.window = deque(maxlen=history_length)
        self.annotation = {'textfield': [], 'coordinates': []}
        
        self.image.append(image[window[0][1]:window[1][1], window[0][0]:window[1][0]])
        self.prediction.append(prediction)

        self.window.append(window)
        
        for key in scalers:
            Detection._scalers[key] = scalers[key]

        
        self.annotation['textfield'] = 'Prediction confidence: {:.0f} %'.format(100*prediction)
        
        self._feature_vector = deque(maxlen=history_length)
        
    def imshow(self):
        plt.title(self.annotation['textfield'])
        plt.imshow(self.image[-1])
        
    def update(self, image, window, prediction):
        self.mean_window(window)
        self.image.append(image[window[0][1]:window[1][1], window[0][0]:window[1][0]])
        self.prediction.append(prediction)
        self.annotation['textfield'] = "Prediction confidence: {:.0f} %".format(100*prediction)
        
        
    def mean_window(self, new_window):
        self.window.append(new_window)
        w = np.array(self.window).reshape(-1, 4)
        mw = np.mean(w, axis=0).astype(int)
        self.window[-1] = ((mw[0], mw[1]), (mw[2], mw[3]))
        
    @property
    def feature_vector(self):
        w =  self.window[-1]
        feature_vector = []
        if (w[0][0] != w[1][0]) & (w[0][1] != w[1][1]):
            image_scaled = cv2.resize(self.image[-1], (64, 64)) 
        
            feature_vector = pl.extract_features(image_scaled, Detection._scalers, **Detection._kw_extract)
        return feature_vector
    @property
    def center_coords(self):
        (x1, y1), (x2, y2) = self.window[-1]
        width = x2-x1
        height= y2-y1
        xc = (int)(x1+width/2)
        yc = (int)(y1+height/2)
        return xc, yc   

class Car(Detection):
    
    _car_ids = count(0)
    def __init__(self, image, window, prediction, scalers):
        super(self.__class__, self).__init__( image, window, prediction, scalers )

        self.id = next(self._car_ids)
        self.car_lost_counter    = 0
        
    @property
    def local_scan_windows(self):
        windows = []
        xc, yc = self.center_coords
        width_s  = 200

        height_s = width_s * 0.75  
        width_e  = 100
        height_e = width_e * 0.75  
        yc_s, yc_e   = (700, 450)
        nb_pixel     = (yc_s - yc_e)
        t = 1 - (yc_s - yc)/nb_pixel
        
        width  = width_s  * t + width_e  * (1-t)
        height = height_s * t + height_e * (1-t)
    
        overlap = Detection._overlap        
        scaling_factor = Detection._scaling_factor

        xy_area_size = (width*scaling_factor, height*scaling_factor)

        nb_windows_width = (int)((xy_area_size[0]/width - 1)*1/overlap + 1)
        nb_windows_height = (int)((xy_area_size[1]/height - 1)*1/overlap + 1)

        dx      = xy_area_size[0]/2  - width/2  - np.mod(xy_area_size[0], width*overlap)/2
        dy      = xy_area_size[1]/2  - height/2 -  np.mod(xy_area_size[1], height*overlap)/2


        coords_x  = np.linspace(xc-dx, xc+dx, nb_windows_width)
        coords_y  = np.linspace(yc-dy, yc+dy, nb_windows_height)
        X, Y = np.meshgrid(coords_x, coords_y)

        for x,y in zip(X.flatten(), Y.flatten()):
            windows.append(pl.rectangle((int)(x), (int)(y), width, height))
        return windows
        
         
    def feature_distance(self, feature_matrix):
        pass
    
    def spatial_distance(self, center_coordinates):
        pass
