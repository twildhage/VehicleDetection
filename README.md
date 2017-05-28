
[//]: # (Image References)
[image1]: ./results/full_scan_heatmap.png
[image2]: ./results/full_scan_windows.png
[image3]: ./results/hog_images.png
[image4]: ./results/local_scan_windows.png
[image5]: ./results/predictions_16x.png
[video1]: ./project_video.mp4



**Vehicle Detection Project**
------------------------------
## Introduction
The goal of this project is to train a classifier on vehicle and non-vehicle images and use it to detect vehicles in a video stream.
There are two main approaches to this problem: classic computer vision techniques and modern deep learning neural networks.
I decided to try out both approaches to get a better feeling how the two techniques compare. The classic approach worked really well for the given task.
However, I decided to spend more time with the deep learning model to get more experienced in applying these techniques.
Therefore I'll explain those results here.

## Model Design

Although one could use a pretrained model like VGG and fine tune this on the given data set, this would likely be an overkill for this project.
Therefore I started from scratch with a new network. The first network had about 1 million parameters and reached a 98% accuracy on the test set.
However, it took rather long to both train the model and use it on the video frames.
After some iterations on the model design it turned out, that, for the given problem, this was still too large and a much smaller model would performed equally well. Reducing the number of kernels drastically (from 256 to 32) and replacing the fully connected layer with a convolutional layer as output, resulted in a model with just about 35 thousand parameters.
Here is the implementation of the keras-model:
```python
init = Input(shape=(64, 64, 3))
# Convolutional layers
x    = Conv2D(16, (3,3), activation='relu', padding='same', strides=(2,2) )(init)
x    = Conv2D(32, (3,3), activation='relu', padding='same', strides=(1,1) )(x)
x    = Dropout(0.4)(x)
x    = Conv2D(32, (3,3), activation='relu', padding='same', strides=(2,2) )(x)
x    = Conv2D(64, (3,3), activation='relu', padding='same', strides=(2,2) )(x)
x    = Dropout(0.6)(x)
x    = Conv2D(1, (8,8), activation='sigmoid', padding='valid', strides=(1,1) )(x)
out    = Flatten()(x)   
model = Model(init, out)
```
The structure of the model is very simple:

    Layer (type)                 Output Shape              Param #   
    ----------------------------------------------------------------
    input_1 (InputLayer)         (None, 64, 64, 3)         0         
    conv2d_1 (Conv2D)            (None, 32, 32, 16)        448       
    conv2d_2 (Conv2D)            (None, 32, 32, 32)        4640      
    dropout_1 (Dropout)          (None, 32, 32, 32)        0         
    conv2d_3 (Conv2D)            (None, 16, 16, 32)        9248      
    conv2d_4 (Conv2D)            (None, 8, 8, 64)          18496     
    dropout_2 (Dropout)          (None, 8, 8, 64)          0         
    conv2d_5 (Conv2D)            (None, 1, 1, 1)           4097      
    flatten_1 (Flatten)          (None, 1)                 0         
    Total params: 36,929
    Trainable params: 36,929
    Non-trainable params: 0

## Training
The model was then trained as follows:

    batch size        256
    epochs            20
    steps per epoch   100

and reached a 99.4% accuracy on the test set.

## Predictions
I used the model to make some predictions.
Here are a few images and the corresponding labels and predictions taken from the validation set.
It can be observed that most of the images are classified accurately. However, there are also examples of false predictions. The car at position (row:1 col:4) for example, is predicted has a classification score of only 15 %.
The image of the car is rather dark. This could be an indication, that one could improve the model's accuracy by augmenting the brightness of the images while training.
![alt text][image5]
## Pipeline
The pipeline is build around the use case of detecting vehicles in a video stream in real time.
The deep learning approach allows to use the raw images. The only preprocessing step that was necessary in my case, was to scale the images from [0, 255] to a [0, 1] range since the training was also done for this range.
I had to learn the hard way that the size, shape and overlap of the search window had a strong influence on the overall performance of the detection pipeline.
A brute force approach to detect vehicles would be to scan the entire screen for each frame. I decided use mainly local scans and perform full scans only every 30 frames, so roughly one time per second and.
For the full scan the street is divided into different areas:
* near range (red)
* mid range (green)
* far range (blue)
* neighbor range (white)

The search windows are created smaller for areas closer to the horizon. The largest search windows, are for neighboring vehicles, which in our case, are on the right side of the video frames. The width-height ratios of the windows are kept equal for the near, mid and far range windows. The neighbor windows are squares.
The following image gives an impression on the different window sizes.
![alt text][image2]
The left side of the road is not scanned here. Of course, the left lane would also be scanned in a real application.
The heatmap of the windows shows how many layers of search windows result from the different windows grids:
![alt text][image1]
It can be seen, that a maximum of almost 150 layers results from this window grid.
While this worked fine in this project, I'm sure that there is a lot of room for improvement here...
The detection ran with an average of 10 frames/sec. on my PC.


Between the full scans only local scans are performed. The scanning is done on areas where a car has been detected in the previous frame.
This increases the performance and allows to improve the detection accuracy by adjusting the search window size according to the position of the vehicle.
The window size is decrease continuously as the car moves further towards the horizon. The image below shows some examples of local search areas:
![alt text][image4]
The areas here are created with no overlap so that the window size can be seen clearly. The magenta colored fine meshed area on the right shows a search area with an overlap of 80% as it is applied throughout  this project.
### Search Logic
As mentioned in the previous section I used full scans and local scans.
Two types of classes were created for the detection pipeline.

**Classes**:
* **Detector**: This class serves as a container for different types of objects. In this project only one type is implemented: Cars. The Detector object has methods to perform the the full and the local scans.
* **Detections**: This is the base class of all detectable objects. This could be cars, trucks, pedestrians, bicycles, ... For this project only Cars are implemented.
* **Cars**: This class contains, in addition to the Detections class, a car-ID which, in principle should be unique for any detected car. However, the steps to distinguish between different cars, could not be finished in this project. The idea was to use a feature-vector a determine the distance in feature-space as a measure to decide if two images belong to the same car.

The following pseudo code describes how the classes and the two search types work together:

    Init:
    perform full scan
    Run:
    calculate distance of new detections to all cars that are already detected
    if no cars detected yet:
      new detections become detected cars
    else if some cars have already been detected:
      calculate distance of new detections to each car
        if distance to small:
          delete detections
        else:
          detection becomes detected cars
    perform local scan and update car attributes
    if car lost for too long:
      delete car
    if timer > 30:
      perform full scan
    repeat Run until end of video stream

## Results
The resulting video can be found [here](./project_video,mp4) and on [Youtube](https://youtu.be/yZJy-53lmVo).

The classification model and the pipeline were able to detect all cars in the video.
However, here is a list of some shortcomings:
* The detector can not distinguish between different cars. Therefore, matched windows are mixed up when cars pass each other (video at 32 sec.)
* The size of the search windows is not based on perspective. Therefore, the window sizes are not consistent with the car sizes at all distances.
* in different lighting scenes cars are not always detected accurately (video at 41 sec.)
* The left line is not scanned. This is not a problem for this project, but has to be possible in a real application. Scanning a larger area with more cars, strongly influences the performance.

## Discussion
Here are some ideas how the detection could be improved:
* Different training method: In this project I used a deep neural net to detect vehicles in a video stream. But the training was perform in a classical manner. A different approach would be to use a data set with already boxed objects and let the network directly learn the box locations.
* Data augmentation: Adjusting the brightness or flipping and rotation the images can increase the training set and improve generalization.
* Add feature-vectors to each detected object to distinguish them from each other
* Calculate the velocity vector of the car estimate the location for the next search area
* Use the velocity estimation to fill the gab, when the car is currently not visible (behind some car). This requires to distinguish between different cars.
