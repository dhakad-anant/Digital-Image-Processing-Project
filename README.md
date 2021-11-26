# Digital Image Processing(CS517) Project

Seam carving is a content-aware image resizing algorithm.It works by finding a number of seams, or paths of least importance, in an image and automatically removing them to reduce image dimensions or inserting them to extend the image. Seam carving also includes the ability to delete entire items from images and the capacity to manually define areas in which pixels may not be edited.

## Interface 

<img src="https://github.com/dhakad-anant/Digital-Image-Processing-Project/blob/main/interface.jpeg" height="" width="500">

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
```
python seam_carving.py (-resize | -remove) -im <IM_PATH> -out <OUTPUT_IM_NAME> 
                       [-mask <MASK_PATH>] [-rmask <REMOVAL_MASK_PATH>] [-dy <DY>] [-dx <DX>] 
                       [-vis] [-hremove] [-backward_energy]
```

The program is run via the command-line. There are two modes of operations: `resize` or `remove`. The former is for resizing an image vertically or horizontally and the latter is for removing an object as specified by a mask.

For both modes:
* `-im`: The path to the image to be processed.
* `-out`: The name for the output image.
* `-mask`: (Optional) The path to the protective mask. The mask should be binary and have the same size as the input image. White areas represent regions where no seams should be carved (e.g. faces).
* `-vis`: If present, display a window while the algorithm runs showing the seams as they are removed.
* `-backward_energy`: If present, use the backward energy function (i.e. gradient magnitude) instead of the forward energy function (default).

For resizing:
* `-dy`: Number of horizontal seams to add (if positive) or subtract (if negative). Default is 0.
* `-dx`: Number of vertical seams to add (if positive) or subtract (if negative). Default is 0.

For object removal:
* `-rmask`: The path to the removal mask. The mask should be binary and have the same size as the input image. White areas represent regions to be removed.
* `-hremove`: If present, perform seam removal horizontally rather than vertically. This will be more appropriate in certain contexts.
