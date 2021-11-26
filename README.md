# Digital Image Processing(CS517) Project

Seam carving is a content-aware image resizing algorithm.It works by finding a number of seams, or paths of least importance, in an image and automatically removing them to reduce image dimensions or inserting them to extend the image. Seam carving also includes the ability to delete entire items from images and the capacity to manually define areas in which pixels may not be edited.

## Interface 

<img src="https://github.com/dhakad-anant/Digital-Image-Processing-Project/blob/main/interface.jpeg" height="300" width="500">


## Requirements
* OpenCV
* scipy
* numba
* numpy
* django

## Features 

* Input image can be resized without any important information. 
  
* You can manually remove parts/objects in the image.

## Usage

* Choose Action 
    - Resize : for resizing the image.
    - Remove : for removing portions/objects

* Choose Input Image : upload input image 
  
* Name of Output Image : give a suitable output name 

* Change in Height Required : enter height that you want to change( +ve values means expanding the height, and -ve meaning shrinking the height) 
  
* Change in Width Required : enter Width that you want to change( +ve values means expanding the Width, and -ve meaning shrinking the Width) 

* Visualize Seam Removal Process : if you want to see the whole process turn it on (select yes)
  
* Choose Energy Function : select desired option.
  
## Note

* When choosing remove option: 
    - after clicking on the submit button a dailog box with image will appear on your screen.
    - scribble the area with mouse-click that you want to remove. 
    - When the part is selected click "s" on your keyword. 
    - Wait for few seconds and image will start removing the described object.
  