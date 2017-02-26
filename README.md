## Udacity SDC Nanodegree Project  5 - Vehicle Detection
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

#### The following sections address each of the rubric points for the project and describe my approach for tackling each of them.

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. 

Provided below.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The section titled “P5-Functions” in the IPython Notebook (Project5.ipynb) contains each of the functions that I used to process images and train my classifier. After reading in all the available training images (8035 vehicle images and 8020 non-vehicle images), I then perform various operations to extract features from the images. The functions “get_hog_features” and “find_cars” are what I used to extract HOG features from the images. Using the get_hog_features function, I have shown below some images of HOG features from the “vehicles” and “non-vehicles” images that comprise the training set. 

There are many parameters that can be adjusted for the `skimage.hog()` HOG features tool. After some experimentation, I stuck with the baseline parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` as I did not see much improvement in the accuracy of my linear SVM classifier when I varied these parameters. 

My main tuning for the HOG features was the color spaces and the number of channels used. The HOG features for each channel for HSV, YCrCb and LUV color channels can be seen below.

#### 2. Explain how you settled on your final choice of HOG parameters.

The following table shows the different combinations of color spaces and channels that I tried and how the test accuracy of the classifier varied with each. I found that the LUV color space worked the best in producing and accurate and reliable solution. I also used all three channels from the LUV color space because less false positives were produced when testing on video. Even though the change in accuracy from 1 to “ALL” channels was fairly small, I actually found that it made a significant difference when testing on the video stream. Unfortunately, using all three channels does slow the solution down. 


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The “Train SVM Classifier” section of the P5-Vehicle.ipynb notebook is where I trained the linear SVM classifier. After the images are read in, HOG features, spatial features and color histogram features are all extracted from the training and test images. I experimented with not including the spatial and color histogram features, but found that the classifier was less likely to produce false positives with both of these types of features included. The table below shows the classifier accuracy using only HOG features, HOG + spatial features and HOG + spatial + histogram features. These features are then normalized, shuffled and split into training and test sets before they are presented to the SVM.

The SVM is a large margin classifier, which means that it is attempting to create linear boundaries between the non-car and car features that maximize the margin. The final accuracy of my linear SVM on the test set was 99.4%.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I created a separate section in the notebook entitled “Sliding Windows” where I could test the size, location and number of sliding windows. After a lot of experimentation I found that I could achieve a successful result with 5 different size windows. The largest windows searched the bottom of the images, whereas the smaller windows searched closed to the horizon line. This corresponds to the perspective size of the vehicles in the image. I also used rectangles rather than boxes because of the typical shape of the vehicles. A table of the sliding window sizes is shown below and an image with all of these windows is also shown. Finally I show an image that uses the sliding windows with the SVM classifier together to determine car/non-car areas of the image.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Once I had trained a fairly robust classifier and tuned my sliding windows approach, I then used the heat map approach to help with the false positives and create well positioned detection boxes. The functions used to perform these operations are in the section entitled “Heat Map for Final detection”. The heatmap function assigns a +1 for the pixel locations where there has been a positive vehicle detection by the classifier. Typically when a vehicle actually exists (true positive) multiple detections will be made due to overlapping nature and varying scale of the sliding windows. Therefore a thresholding can be applied to filter out the detections where only a single window was classified as a vehicle (usually a false positive). The below images show some examples after the heat map thresholding is performed. Still not perfect, but getting closer! 

IMAGES

While this worked fairly well, it was clear that when using this pipeline on video, multiple consecutive frames could be used to improve the results. These techniques are described in the following sections.
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video.mp4) for the vehicles detections.

Here is another link to the combined results of my P4 and P5 approaches.

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The video implementation was a little different as it allowed for the use of multiple consecutive image frames. From each frame I stored the heat map and the detection boxes that were produced using the “draw on image” function. By storing these values, it allowed me to combine the results from multiple frames. For the heat maps, I summed the results from the most recent 10 frames. This allowed me to increase the heatmap threshold. With this integrated heat map technique I saw several positive outcomes.  The first was that the false positives were reduced, the second was that the true positives increased (missing a vehicle in a single frame was okay as long as the next frames picked it up) and finally, the detection boxes became smoother. Once the final heat maps are determined, the `scipy.ndimage.measurements.label()` function was used to identify individual vehicles in the heat map.

The second technique that I used in the video processing was averaging and outlier removal for the detection boxes drawn on the image. I took an average of the box corners from the previous 5 frames and if the current frame did not vary by more than 50% from that average, then it was included in a new average, otherwise it was discarded.

Here's an example result showing the summed heat map from a series of frames of video and the resulting detection boxes. 

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The biggest problems that I faced were false positive detections and variability of the vehicle detection boxes from frame to frame. These issues were improved using the following techniques:

1. Changing the colorspace to LUV which improved the classifier accuracy to 99.4% on the test images.
2. Tuning of the number, size and locations of the sliding windows had a significant impact on the results.
3. Integrating the heat maps over several frames so that the vehicles showed up strongly enough that the threshold could be increased.
4. Averaging of the box corners over several frames made them significantly smoother.

While this approach works fairly well on the project video, it will still fail in many situations. The classifier was only trained on approximately 16000 images, which could be significantly increased to help the model generalize. Changes in lighting and shadows or locations of color saturation will still cause the model to make false positive detections. This could be potentially improved by using augmentations (other colorspaces, brightness adjustments, blurring, jittering) to the training images. 

With more time I would be very curious to train a Convolutional Neural Network that uses the raw images rather than HOG features to classify the images. My approach for this project does not perform fast enough for real time and I would be interested to test the speed of other approaches running on a state of the art GPU to see if they could meet the requirements of performing in a real vehicle. 

Now that I have my techniques from P4 and P5 combined together my next step is to take some video from my own car and see if these techniques can work in the real world!

