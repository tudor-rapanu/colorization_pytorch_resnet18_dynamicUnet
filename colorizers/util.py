from PIL import Image
import numpy as np
from skimage import color
import torch
import torch.nn.functional as F
from IPython import embed

    # out_np = image matrix converted to numpy array 
def load_img(img_path):
	out_np = np.asarray(Image.open(img_path))
	if(out_np.ndim==2):
		out_np = np.tile(out_np[:,:,None],3)
	return out_np


    # Image.fromarray(img) shows the actual size of the image and the color mode (RGB here)
    # and creates an image memory from an object exporting the array interface
    # Image.fromarray(img).resize((HW[1],HW[0]), resample=resample) returns a resized copy of the image
    # size - the requested size in pixels: 256x256 
    # resample - an optional resampling filter: here it is 3
    # np.asarray transforms the image memory to a numpy array that is later used for processing
def resize_img_resample(img, HW=(256,256), resample=3):
	return np.asarray(Image.fromarray(img).resize((HW[1],HW[0]), resample=resample))


    # img_rgb_orig is the original image matrix in RGB color space
    # img_rgb_rs is the resized image matrix in RGB color space
    # HW is the height and width of the resized image - here is 256x256
    # img_lab_orig is the original image matrix converted to Lab color space 
    # img_lab_rs is the resized image matrix converted to Lab color space 
    # img_l_orig and img_l_rs catch the L component of the respective matrix
    # A torch.Tensor is a multi-dimensional matrix containing elements of a single data type.
    # With tens_orig_l and tens_rs_l the function returns original size L and resized L as torch Tensors
    #used later for post processing the original and resized images
def preprocess_img(img_rgb_orig, HW=(256,256), resample=3):
	img_rgb_rs = resize_img_resample(img_rgb_orig, HW=HW, resample=resample) # img_rgb_rs size is  256x256x3
	   
	img_lab_orig = color.rgb2lab(img_rgb_orig)
	img_lab_rs = color.rgb2lab(img_rgb_rs)

	img_l_orig = img_lab_orig[:,:,0]
	img_l_rs = img_lab_rs[:,:,0]

	tens_orig_l = torch.Tensor(img_l_orig)[None,None,:,:]
	tens_rs_l = torch.Tensor(img_l_rs)[None,None,:,:]

	return (tens_orig_l, tens_rs_l)


    # The function returns the result converted back to RGB and with the size if the orginal image 
    # HW_orig and HW only retrive the sizes of the original and resized images
    # tens_orig_l is the tensor of the L component in the original image
    # out_ab is the tensor for the ab channels returned either by the colorization models or the 0 filled matrix (b&w image)
    # F.interpolate will upsample the input (here, the out_ab Tensor) to the specified size (here, HW_orig)
    # The algorithm used for interpolation is determined by mode (here, bilinear)
    # out_ab_orig is the result of the upsampling mentioned before
    # torch.cat concatenates the given sequence of seq tensors in the given dimension
    # out_lab_orig is the result of the concatenation mentioned before 
    # Result is converted to a numpy array, then cut one dimension, then transposing it to HxWx3 dimensions
    # Finally, before returning, result is converted back to RGB - operations are done on CPU
def postprocess_tens(tens_orig_l, out_ab, mode='bilinear'):
	# tens_orig_l 	1 x 1 x H_orig x W_orig
	# out_ab 		1 x 2 x H x W

	HW_orig = tens_orig_l.shape[2:]
	HW = out_ab.shape[2:]

	# call resize function if needed
	if(HW_orig[0]!=HW[0] or HW_orig[1]!=HW[1]):
		out_ab_orig = F.interpolate(out_ab, size=HW_orig, mode='bilinear', align_corners=False)
	else:
		out_ab_orig = out_ab

	out_lab_orig = torch.cat((tens_orig_l, out_ab_orig), dim=1)
	return color.lab2rgb(out_lab_orig.data.cpu().numpy()[0,...].transpose((1,2,0)))

