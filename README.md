# FacialKeyPoints
The core of this project is a cutom CNN that extracts facial keypoints. The data was trained on
[YouTube Faces Dataset](https://www.cs.tau.ac.il/~wolf/ytfaces/), after being put through a
set of preprocessing transforms.

Haar cascades are used to detect faces and extract them from images. Those ROIs are then
run through the CNN to detect facial keypoints and display them in an overlay,
 with keypoints and/or sunglasses.

# Dependencies
numpy
opencv
pytorch

# Usage / Getting Started
If you don't have the dataset, you may just want to play around with live camera overlays
Make sure a camera is connected, and run LiveFaceSearch.py 

To explore the preprocessing, model creation, and training process, look through the notebooks.
These notebooks were created as the final project for Udacity's Intro to Computer Vision Course (nd891),
and gave me the idea for this project!

# Future Improvements / Known Issues
- The youtube dataset is generally taken under good lighting conditions, so very directional lighting
 will skew detection away from the dark patches a bit.
- Representation of exagerated facial expressions is fairly limited. While it captures features
quite well, heavily contorted / funny faces are under-represented. I find the work a little,
but the model has overfit the standard face ratios.
- Haar Cascades do well with faces that look stright-on. Profiles are not picked up as well.