import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import os
from glob import glob


def print_image_properties(image):
    print("Shape:\t\t", image.shape)
    print("Max, Min: \t[{0:.2f}, {1:.2f}]".format(image.max(), image.min()))

def get_image_files(path):
    image_folders     = os.listdir(path)
    image_files = {key: [] for key in image_folders}
    for folder in image_folders:

        subfolders = os.listdir(path+folder)
        print("Subfolders of {}:".format(folder))    
        print(subfolders)

        for subfolder in subfolders:
            abs_path = path + folder + '/' + subfolder + '/*.png'
            if not subfolder.startswith('.'):
                files =  glob(abs_path)
                [image_files[folder].append(file) for file in files]
    return image_files


def get_shuffle_split_train_test_datasets(features, nb_samples=10, test_size=0.2):
    X = {'train': [], 'test': []}
    y = {'train': [], 'test': []}

    X_stack = np.vstack([features['vehicles'][0:nb_samples], features['non-vehicles'][0:nb_samples]])
    y_stack = np.vstack([np.ones((nb_samples, 1)), np.zeros((nb_samples, 1))])

    X['train'], X['test'], y['train'], y['test'] = train_test_split(X_stack, y_stack,
                                                                    test_size=test_size, random_state=42)

    return X, y


def get_shuffle_split_train_test_images(image_files, nb_samples=10, test_size=0.2):
    files = {'train': [], 'test': []}
    images = {'train': [], 'test': []}

    image_files_stack = np.vstack([image_files['vehicles'][0:nb_samples],
                                   image_files['non-vehicles'][0:nb_samples]])
    indices           = np.arange(0, len(image_files_stack))

    
    files['train'], files['test'], _, _ = train_test_split(image_files_stack, indices,
                                                                    test_size=test_size, random_state=42)
    
    for key in files:
        for file in files[key][0]:
            images[key].append(mpimg.imread(file))
    
    return images, files

            

# Define a function to return HOG features and visualization
def get_hog_features(image, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(image, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(image, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

# Define a function to compute binned color features  
def bin_spatial(image, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(image, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features 
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(image, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(image[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(image[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(image[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(image_files, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = {'spatial': [], 'hist': [], 'hog': []}
    # Iterate through the list of images
    for file in image_files:
        image_features    = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      

        if spatial_feat == True:
            spatial_feature = bin_spatial(feature_image, size=spatial_size)
            features['spatial'].append(spatial_feature)
        if hist_feat == True:
            # Apply color_hist()
            hist_feature = color_hist(feature_image, nbins=hist_bins)
            features['hist'].append(hist_feature)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_feature = []
                for channel in range(feature_image.shape[2]):
                    hog_feature.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                features['hog'].append(np.ravel(hog_feature))        
            else:
                features['hog'].append(get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True))
            # Append the new feature vector to the features list
    f = []
    for key in features:
        if features[key]:
            X = np.vstack(features[key]).astype(np.float64)                        
            # Fit a per-column scaler
            X_scaler = StandardScaler().fit(X)
            # Apply the scaler to X
            scaled_X = X_scaler.transform(X)
            f.append(scaled_X)
            print(scaled_X.shape)
    result =  np.concatenate(f, axis=1)
    # Return list of feature vectors
    return result    
# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(image, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = image.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = image.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

# Define a function to draw bounding boxes
def draw_boxes(image, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(image)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

