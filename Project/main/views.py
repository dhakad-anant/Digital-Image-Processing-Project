from django.shortcuts import get_object_or_404, render, redirect
from django.http import Http404
from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.core.files.storage import FileSystemStorage
import numpy as np
import cv2
from numba import jit
from scipy import ndimage as ndi
import math
from scipy.ndimage.filters import convolve
from .classes.utils_classes import Sketcher
import os

SEAM_COLOR = np.array([0, 0,0])    # seam visualization color (BGR)
SHOULD_DOWNSIZE = True                  # if True, downsize image for faster carving
DOWNSIZE_WIDTH = 500                      # resized image width if SHOULD_DOWNSIZE is True
ENERGY_MASK_CONST = 100000.0              # large energy value for protective masking
MASK_THRESHOLD = 250                       # minimum pixel intensity for binary mask
USE_FORWARD_ENERGY = True 

########################################
# UTILITY CODE
########################################

def makemask(im):
    """
    You can select object/portion to carve out

    Keys:
    s     - save the masked the image
    SPACE - reset the inpainting mask
    ESC   - exit
    """
    if im is None:
        print("\tEmpyt of invalid input image in function")
        return
 
    # Create an image for sketching the mask
    scribble_im = im.copy()
    sketch = Sketcher('Image', [scribble_im], lambda : ((255, 255, 255), 255))
 
    # Sketch a mask
    while True:
        ch = cv2.waitKey()
        if ch == 27: # ESC - exit
            break
        if ch == ord('s'): # s - save the masked the image
            break
        if ch == ord(' '): # SPACE - reset the inpainting mask
            scribble_im[:] = im
            sketch.show()
 
    # define range of white color in HSV
    lower_white = np.array([255, 255, 255])
    upper_white = np.array([255, 255, 255])

    # Create the mask
    mask = cv2.inRange(scribble_im, lower_white, upper_white)

    return mask

def visualize(im, boolmask=None, rotate=False):
    vis = im.astype(np.uint8)
    if boolmask is not None:
        vis[np.where(boolmask == False)] = SEAM_COLOR
    if rotate:
        vis = rotate_image(vis, False)
    cv2.imshow("Visualization", vis)
    cv2.waitKey(1)
    return vis

def resize(image, width):
    dim = None
    h, w = image.shape[:2]
    dim = (width, int(h * width / float(w)))
    return cv2.resize(image, dim)

def rotate_image(image, clockwise):
    k = 1 if clockwise else 3
    return np.rot90(image, k)    

########################################
# ENERGY FUNCTIONS
########################################

@jit
def energy_one(img):
    filter_du = np.array([
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -1.0],
    ])
    # This converts it from a 2D filter to a 3D filter, replicating the same
    # filter for each channel: R, G, B
    filter_du = np.stack([filter_du] * 3, axis=2)

    filter_dv = np.array([
        [1.0, 0.0, -1.0],
        [2.0, 0.0, -2.0],
        [1.0, 0.0, -1.0],
    ])
    # This converts it from a 2D filter to a 3D filter, replicating the same
    # filter for each channel: R, G, B
    filter_dv = np.stack([filter_dv] * 3, axis=2)

    img = img.astype('float32')
    convolved = np.absolute(convolve(img, filter_du)) + np.absolute(convolve(img, filter_dv))

    # We sum the energies in the red, green, and blue channels
    energy_map = convolved.sum(axis=2)

    return energy_map

def energy_two(im):
    """
    Simple gradient magnitude energy map.
    """
    xgrad = ndi.convolve1d(im, np.array([1, 0, -1]), axis=1, mode='wrap')
    ygrad = ndi.convolve1d(im, np.array([1, 0, -1]), axis=0, mode='wrap')
    
    grad_mag = np.sqrt(np.sum(xgrad**2, axis=2) + np.sum(ygrad**2, axis=2))

    return grad_mag

@jit
def energy_three(im):
    h, w = im.shape[:2]
    im = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)

    energy = np.zeros((h, w))
    m = np.zeros((h, w))
    
    U = np.roll(im, 1, axis=0)
    L = np.roll(im, 1, axis=1)
    R = np.roll(im, -1, axis=1)
    
    cU = np.abs(R - L)
    cL = np.abs(U - L) + cU
    cR = np.abs(U - R) + cU
    
    for i in range(1, h):
        mU = m[i-1]
        mL = np.roll(mU, 1)
        mR = np.roll(mU, -1)
        
        mULR = np.array([mU, mL, mR])
        cULR = np.array([cU[i], cL[i], cR[i]])
        mULR += cULR

        argmins = np.argmin(mULR, axis=0)
        m[i] = np.choose(argmins, mULR)
        energy[i] = np.choose(argmins, cULR)
    
        
    return energy

########################################
# SEAM HELPER FUNCTIONS
######################################## 

@jit
def add_seam(im, seam_idx):
    h, w = im.shape[:2]
    output = np.zeros((h, w + 1, 3))
    for row in range(h):
        col = seam_idx[row]
        for ch in range(3):

            col_min = max(col - 1, 0)
            col_max = min(w - 1, col + 1)

            p = np.average(im[row, col_min: col_max + 1, ch])
            output[row, : col, ch] = im[row, : col, ch]
            output[row, col, ch] = p
            output[row, col + 1:, ch] = im[row, col:, ch]

    return output

@jit
def add_seam_grayscale(im, seam_idx):
    """
    Add a vertical seam to a grayscale image at the indices provided 
    by averaging the pixels values to the left and right of the seam.
    """    
    h, w = im.shape[:2]
    output = np.zeros((h, w + 1))
    for row in range(h):
        col = seam_idx[row]

        col_min = max(col - 1, 0)
        col_max = min(w - 1, col + 1)

        p = np.average(im[row, col_min: col_max + 1])
        output[row, : col] = im[row, : col]
        output[row, col] = p
        output[row, col + 1:] = im[row, col:]

    return output

@jit
def remove_seam(im, boolmask):
    h, w = im.shape[:2]
    boolmask3c = np.stack([boolmask] * 3, axis=2)
    return im[boolmask3c].reshape((h, w - 1, 3))

@jit
def remove_seam_grayscale(im, boolmask):
    h, w = im.shape[:2]
    return im[boolmask].reshape((h, w - 1))

@jit
def get_minimum_seam(im, energyfn, rmask=None):
    """
    DP algorithm for finding the seam of minimum energy. Code adapted from 
    https://karthikkaranth.me/blog/implementing-seam-carving-with-python/
    """
    h, w = im.shape[:2]
    M = energyfn(im)

    if rmask is not None:
        M[np.where(rmask > MASK_THRESHOLD)] = -ENERGY_MASK_CONST * 1000

    backtrack = np.zeros_like(M, dtype=int)

    # populate DP matrix
    for y in range(1, h):
        for x in range(w):

            min_x = max(x - 1, 0)
            max_x = min(w - 1, x + 1)
            
            min_prev_seam_energy = math.inf
            back_pointer = -1
            for ener in range(min_x, max_x + 1):

                if(M[y - 1, ener] < min_prev_seam_energy):
                    back_pointer = ener
                min_prev_seam_energy = min(min_prev_seam_energy, M[y - 1, ener])

            M[y, x] += min_prev_seam_energy
            backtrack[y, x] = back_pointer

    # backtrack to find path
    seam_idx = []
    boolmask = np.ones((h, w), dtype=bool)

    minimum_seam = math.inf
    index_min_seam = -1
    for x in range(w):

        if(minimum_seam > M[h - 1, x]):
            index_min_seam = x
        minimum_seam = min(minimum_seam, M[h - 1, x])
    
    for i in range(h-1, -1, -1):
        boolmask[i, index_min_seam] = False
        seam_idx.append(index_min_seam)
        index_min_seam = backtrack[i, index_min_seam]

    seam_idx.reverse()
    return np.array(seam_idx), boolmask

########################################
# MAIN ALGORITHM
######################################## 

def seams_removal(im, num_remove, energyfn, vis=False, rot=False):
    for _ in range(num_remove):
        seam_idx, boolmask = get_minimum_seam(im, energyfn)
        if vis:
            visualize(im, boolmask, rotate=rot)
        im = remove_seam(im, boolmask)
    return im


def seams_insertion(im, num_add, energyfn, vis=False, rot=False):
    seams_record = []
    temp_im = im.copy()

    for _ in range(num_add):
        seam_idx, boolmask = get_minimum_seam(temp_im, energyfn)
        if vis:
            visualize(temp_im, boolmask, rotate=rot)

        seams_record.append(seam_idx)
        temp_im = remove_seam(temp_im, boolmask)

    seams_record.reverse()

    for _ in range(num_add):
        seam = seams_record.pop()
        im = add_seam(im, seam)
        if vis:
            visualize(im, rotate=rot)

        # update the remaining seam indices
        for remaining_seam in seams_record:
            remaining_seam[np.where(remaining_seam >= seam)] += 2         

    return im

########################################
# MAIN DRIVER FUNCTIONS
########################################

def seam_carve(im, dy, dx, energyfn, vis=False):
    im = im.astype(np.float64)
    h, w = im.shape[:2]

    if(h + dy <= 0 or w + dx <= 0):
        print("Please provide valid input parameters. Terminating the program...")
        exit(1)

    output = im

    if dx < 0:
        output = seams_removal(output, -dx, energyfn, vis)

    elif dx > 0:
        output = seams_insertion(output, dx, energyfn, vis)

    if dy < 0:
        output = rotate_image(output, True)
        output = seams_removal(output, -dy, energyfn, vis, rot=True)
        output = rotate_image(output, False)

    elif dy > 0:
        output = rotate_image(output, True)
        output = seams_insertion(output, dy, energyfn, vis, rot=True)
        output = rotate_image(output, False)

    return output


def object_removal(im, rmask, energyfn, vis=False, horizontal_removal=False):
    im = im.astype(np.float64)
    rmask = rmask.astype(np.float64)
    
    output = im

    h, w = im.shape[:2]

    if horizontal_removal:
        output = rotate_image(output, True)
        rmask = rotate_image(rmask, True)

    while len(np.where(rmask > MASK_THRESHOLD)[0]) > 0:
        seam_idx, boolmask = get_minimum_seam(output, energyfn, rmask)
        if vis:
            visualize(output, boolmask, rotate=horizontal_removal)            
        output = remove_seam(output, boolmask)
        rmask = remove_seam_grayscale(rmask, boolmask)

    num_add = (h if horizontal_removal else w) - output.shape[1]
    output = seams_insertion(output, num_add, energyfn, vis, rot=horizontal_removal)
    if horizontal_removal:
        output = rotate_image(output, False)
    return output        

# Create your views here.
def seamcarve(request):
    if request.method == "POST":

        action = request.POST.get('action')
        inputimagex = request.FILES['inputimage']

        fr = FileSystemStorage()
        filename = fr.save(inputimagex.name, inputimagex)

        dytemp = request.POST.get('heightchange')
        dxtemp = request.POST.get('widthchange')
        dovis = request.POST.get('dovis')
        energyfunc = request.POST.get('energyfunc')

        urlInputFile = os.getcwd()+'/'+filename
        outputURL = os.getcwd()+'/'+'output_'+filename

        im=cv2.imread(urlInputFile)
        rmask = None
        if(action == "remove"):
            rmask = makemask(im)
            outputMaskURL = os.getcwd() + '/mask_'+filename
            cv2.imwrite(outputMaskURL, rmask) 

        energyfn = None
        if energyfunc == "one":
            energyfn = energy_one
        elif energyfunc == "two":
            energyfn = energy_two
        else:
            energyfn = energy_three

        if im is None:
            print("\tError : Invalid \"im\" file path")
            quit()
        
        # downsize image for faster processing
        h, w = im.shape[:2]
        if SHOULD_DOWNSIZE and w > DOWNSIZE_WIDTH:
            im = resize(im, width=DOWNSIZE_WIDTH)
            if rmask is not None:
                rmask = resize(rmask, width=DOWNSIZE_WIDTH)

        modifiedInputURL = os.getcwd()+'/modified_'+filename
        cv2.imwrite(modifiedInputURL, im)

        tovisualize = False
        if dovis == "Yes":
            tovisualize = True

        # image resize mode
        if action == "resize":
            dy, dx = int(dytemp), int(dxtemp)
            if(dy is None or dx is None):
                print("Please provide dy and dx values. Terminating the program...")
                exit(1)
            output = seam_carve(im, dy, dx, energyfn, tovisualize)
            cv2.imwrite(outputURL, output)
        
        # object removal mode
        elif action == "remove":
            assert rmask is not None
            output = object_removal(im, rmask, energyfn, tovisualize, False)
            cv2.imwrite(outputURL, output)

        cv2.destroyAllWindows()

    return render(request, 'main/homepage.html',{})