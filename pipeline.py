import matplotlib.image as mpimg
import matplotlib.pyplot as plt
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
    print("Type: {}".format(type(image[0,0,0])))

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
                
    dataset = {'files': [], 'labels': []}
    for key in image_files:
        print(key)
        for file in image_files[key]:
            dataset['files'].append(file)
            if key == 'vehicles':
                dataset['labels'].append(1)
            elif key == 'non-vehicles':
                dataset['labels'].append(0)
                
    np.random.seed(42)
    np.random.shuffle(dataset['files'])
    np.random.seed(42)
    np.random.shuffle(dataset['labels'])            
    
    return dataset

def split_train_test_dataset(data, test_size=0.2):
    data_part = {'train': [], 'test': []}
    
    L = len(data)
    idx_split = (int)(L * (1-test_size))
    
    data_part['train'] = data[0:idx_split]
    data_part['test']  = data[idx_split::]

    return data_part


def extract_and_split(dataset, kw_extract, kw_split, nb_samples=100):
    
    features, scalers = extract_features_from_files(dataset['files'][0:nb_samples], **kw_extract)
    labels   = np.array(dataset['labels'][0:nb_samples])
    files    = dataset['files'][0:nb_samples]

    print("\nFeatures: \t{0}\nLabels: \t{1}\nFiles: \t\t{2}\n".format(features.shape, labels.shape, len(files)))

    X = split_train_test_dataset(features, **kw_split)
    y = split_train_test_dataset(labels, **kw_split)
    f = split_train_test_dataset(files, **kw_split)

    print("Number of samples \ntraining:\t{0} \ntest:\t\t{1}".format(X['train'].shape[0], X['test'].shape[0]))
    return X, y, f, scalers
            

# Define a function to return HOG features and visualization
def get_hog_features(image, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(image, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  block_norm='L2-Hys',
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(image, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, 
                       block_norm='L2-Hys',
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
def extract_features_from_files(image_files, kw_spatial, kw_hist, kw_hog,
                               color_space='RGB',
                               hog_channel=0,
                               spatial_feat=True,
                               hist_feat=True,
                               hog_feat=True):

    # Create a list to append feature vectors to
    features = {'spatial': [], 'hist': [], 'hog': []}
    scalers   = {key: StandardScaler() for key in features}
    
    # Iterate through the list of images
    for file in image_files:
        image_features    = []
        # Read in each one by one
        image = mpimg.imread(file)
        
        
        # Scale input
        if type(image[0,0,0]) == np.uint8:
            image = (image / 255).astype(np.float32)
    
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

        # Scale input
        if type(feature_image[0,0,0]) == np.uint8:
            feature_image = (feature_image / 255).astype(np.float32)

        if spatial_feat == True:
            spatial_feature = bin_spatial(feature_image, **kw_spatial)
            features['spatial'].append(spatial_feature)
        if hist_feat == True:
            # Apply color_hist()
            hist_feature = color_hist(feature_image, **kw_hist)
            features['hist'].append(hist_feature)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_feature = []
                for channel in range(feature_image.shape[2]):
                    hog_feature.append(get_hog_features(feature_image[:,:,channel],
                                                        **kw_hog, vis=False, feature_vec=True))
                features['hog'].append(np.ravel(hog_feature))        
            else:
                features['hog'].append(get_hog_features(feature_image[:,:,hog_channel],
                                                        **kw_hog, vis=False, feature_vec=True))
            # Append the new feature vector to the features list
    f = []
    for key in features:
        if features[key]:
            X = np.vstack(features[key]).astype(np.float64)                        
            # Fit a per-column scaler
            scalers[key].fit(X)
            # Apply the scaler to X
            scaled_X = scalers[key].transform(X)
            f.append(scaled_X)
    result =  np.concatenate(f, axis=1)
    # Return list of feature vectors
    return result, scalers


# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def extract_features(img, scalers, kw_spatial, kw_hist, kw_hog,
                          color_space='RGB',
                          hog_channel=0,
                          spatial_feat=True,
                          hist_feat=True,
                          hog_feat=True):    
    # Scale input
    if type(img[0,0,0]) == np.uint8:
        img = (img / 255).astype(np.float32)
    #1) Define an empty list to receive features
    features = {'spatial': [], 'hist': [], 'hog': []}
    
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)      
        
    # Scale input
    if type(feature_image[0,0,0]) == np.uint8:
        feature_image = (feature_image / 255).astype(np.float32)
        
        
    #print(type(img[0,0,0]))
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        features['spatial'].append(bin_spatial(feature_image, **kw_spatial))
        
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        features['hist'].append(color_hist(feature_image, **kw_hist))
        
        #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                                    **kw_hog, vis=False, feature_vec=True))
            features['hog'].append(np.ravel(hog_features))        
        else:
            features['hog'].append(get_hog_features(feature_image[:,:,hog_channel],
                                                    **kw_hog, vis=False, feature_vec=True))
    f = []
#    scalers   = {key: StandardScaler() for key in features}
    for key in features:
        if features[key]:
            X = np.vstack(features[key]).astype(np.float64) 
            # Apply the scaler to X
            scaled_X = scalers[key].transform(X)
            f.append(scaled_X)
    result =  np.concatenate(f, axis=1)
    
    #9) Return concatenated array of features
    return result




# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, classifier, threshold=0.95, verbose=0):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    predictions = []
    textfields   = []
    images = []
    #2) Iterate over all windows in the list
    for window in windows:
        if all(coord > 0 for coord in (window[0][0], window[0][1], window[1][0], window[1][1])):
            #3) Extract the test window from original image
            images.append(cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64)))
        
    k,l,m = images[-1].shape
    search_images = np.array(images).reshape(-1, k, l, m)
    pred = classifier.predict(search_images, batch_size=1)
    for p, window in zip(pred, windows):
        #7) If positive (prediction == 1) then save the window
        if p >= threshold:
            predictions.append(p[0])
            textfields.append('{:.2f}'.format(p[0]))
            on_windows.append(window)
    return on_windows, predictions, textfields


def perspective_window(image_shape, window_size=[420, 60], horrizon=420, x_start=200, overlap=0.3, nb_zooms=5):
    height = image_shape[0]
    width  = image_shape[1]
    offset_x = (width % (window_size[0] * overlap)) / 2
    steps = (int) ((10*width)/ (window_size[0] * overlap))
    print(image_shape, window_size[0], offset_x, steps)
    
    xc_s, yc_s, sc_s = ((width/2), height-(window_size[0]/2)-160, window_size[0])
    xc_e, yc_e, sc_e = (width/2-x_start, horrizon, window_size[1]) 
    windows = []
    for step in range(1, steps):
        for t in np.linspace(0, 1, nb_zooms):
            xc = xc_s * t + xc_e * (1-t)
            yc = yc_s * t + yc_e * (1-t)
            sc = sc_s * t + sc_e * (1-t)
            if (xc+sc/2) < width:
                windows.append(square(xc, yc, sc))
            
        xc_s = xc_s + (int)(sc_s * overlap) 
        xc_e = xc_e + (int)(sc_e * overlap) 
        
    return windows


def full_scan_windows(center_coords, xy_area_size, window_widths, width_height_ratio, overlap=0.5):
    windows = []
    xc, yc = center_coords 
    width, height = window_widths, (int)(window_widths * width_height_ratio)
    
    nb_windows_width = (int)((xy_area_size[0]/width - 1)*1/overlap + 1)
    nb_windows_height = (int)((xy_area_size[1]/height - 1)*1/overlap + 1)
    
    dx      = xy_area_size[0]/2  - width/2  - np.mod(xy_area_size[0], width*overlap)/2
    dy      = xy_area_size[1]/2  - height/2 -  np.mod(xy_area_size[1], height*overlap)/2
    
    coords_x  = np.linspace(xc-dx, xc+dx, nb_windows_width)
    coords_y  = np.linspace(yc-dy, yc+dy, nb_windows_height)

    X, Y = np.meshgrid(coords_x, coords_y)
    for x,y in zip(X.flatten(), Y.flatten()):
        windows.append(rectangle((int)(x), (int)(y), width, height))
    return windows




def rectangle(xc, yc, w, h):
    xc = (int)(xc)
    yc = (int)(yc)
    dh = (int)(h/2)
    dw = (int)(w/2) 
    return ((xc-dw, yc-dh), (xc+dw, yc+dh))


def square(xc, yc, s):
    xc = (int)(xc)
    yc = (int)(yc)
    s = (int)(s)
    h = (int)(s/2)
    w = (int)(s/2 * 1.7) 
    return ((xc-w, yc-h), (xc+w, yc+h))


# Define a function to draw bounding boxes
def draw_boxes(image, bboxes, color=(0, 0, 1), thick=3):
    # Make a copy of the image
    imcopy = np.copy(image)
    # Iterate through the bounding boxes
    if len(bboxes) > 0:
        if isinstance(bboxes[0][0], tuple):
            for bbox in bboxes:
                cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        else:
            cv2.rectangle(imcopy, bboxes[0], bboxes[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    heatmap_copy = heatmap.copy()
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap_copy[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap_copy# Iterate through list of bboxes

def add_weighted_heat(heatmap, bbox_list, weights):
    
    heat_norm = add_heat(heatmap, bbox_list) 
    heat_norm[heat_norm==0] = 1
    # Iterate through list of bboxes
    for box, weight in zip(bbox_list, weights):
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += weight
    

    result = heatmap/heat_norm

    # Return updated heatmap
    return result

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap < threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,1), 6)
    # Return the image
    return img

def extract_labeled_bboxes(labels):
    # Iterate through all detected cars
    boxes = []
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        boxes.append(bbox)
    return boxes


def put_text_annotations(image, textfields, textcoords, font):

    color = (255,255,255)
    for textfield, coordinate in zip(textfields, textcoords):
        result = cv2.putText(image, textfield, coordinate, font, 0.75, color, 2, cv2.LINE_AA)
    return result


def random_gamma_shift(image, mode='random', gamma=1.25):
    """
    A gamma correction is used to change the brightness of training images.
    The correction factor 'gamma' is sampled randomly in order to generated
    an even distrubtion of image brightnesses. This shall allow the model to
    generalize.
    The code is inspired by:
    http://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
    :image:
        Source image as numpy array
    :return:
        Gamma corrected version of the source image
    """
    if mode == 'random':
        gamma_ = np.random.uniform(0.1, 2.5)
    elif mode == 'manual':
        gamma_ = gamma
    else:
        print('mode has to be random or manual')
        return 0
    inv_gamma = 1.0 / gamma_
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def get_random_subset_of_dataset(subset_size, dataset, dataset_category='train'):
    
    images = []
    labels = []
    L = len(dataset['labels'])

    for i in range(0, subset_size):
        r = np.random.randint(0, L)
        img = (mpimg.imread(dataset['files'][r])*255).astype(np.uint8)
        images.append((random_gamma_shift(img)/255.0).astype(np.float32))
        labels.append(dataset['labels'][r])
    
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)
    if dataset_category=='train':
        X,y = X_train, y_train
    elif dataset_category=='validation':
        X,y = X_test, y_test
   
    
    return X,y


def get_random_subset_of_dataset_preload(subset_size, dataset, dataset_category='train'):
    
    images = []
    labels = []
    L = len(dataset['labels'])
    for i in range(0, subset_size):
        r = np.random.randint(0, L)
        images.append(dataset['images'][r])
        labels.append(dataset['labels'][r])
    
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)
    if dataset_category=='train':
        X,y = X_train, y_train
    elif dataset_category=='validation':
        X,y = X_test, y_test
    
    return X,y

def generate_batch_debug(batch_size, dataset, dataset_category):
    """
    This function generates a generator, which then yields a training batch.
    If this sounds confusing, check out this excellent explanation on
    stackoverflow:
    http://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do-in-python
    """
#    while True:
    X_batch = []
    y_batch = []
    images, labels = get_random_subset_of_dataset(batch_size, dataset, dataset_category)

    return np.array(images), np.array(labels)
def generate_batch(batch_size, dataset, dataset_category):
    """
    This function generates a generator, which then yields a training batch.
    If this sounds confusing, check out this excellent explanation on
    stackoverflow:
    http://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do-in-python
    """
    while True:
        X_batch = []
        y_batch = []
        images, labels = get_random_subset_of_dataset_preload(batch_size, dataset, dataset_category)

        yield np.array(images), np.array(labels)


