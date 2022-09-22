import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from pathlib import Path
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import math

hw_dir = Path(__file__).parent

# Load images
img1 = io.imread(hw_dir/'image1.png')
img2 = io.imread(hw_dir/'image2.png')

# Part (a)
W = img1.shape[0]       # = 1001 dots
d = np.array([0.4, 2])  # distances (m)
dpi = 300               # dots per inch

img_size = 0.0254 * (W / dpi)  # physical size in (m)
# checked the result ~0.08475 (m)
print("physical image size is", img_size, "(m).")

tan_angle = math.tan(0.5 * math.pi / 180)
delta = 2 * d * tan_angle   # in (m per degree)
# checked the results ~0.0069815 and 0.0349 (m per degree)
print("delta for different distances:", delta[0],
      "and", delta[1], "(m per degree).")
ppd = delta * W / img_size   # in (dots/pixels per degree)
# checked the results ~82.46 and 412.30 (pixels per degree)
print("PPD for different distances:", ppd[0],
      "and", ppd[1], "(pixels per degree).")


# Part (b)
cpd = 5   # Peak contrast sensitivity location (cycles per degree)

img_freq = cpd * np.reciprocal(ppd)   # in (cycles per pixel)
# checked the results ~0.06 and 0.012 (cycles per pixel)
print("image frequencies for different distances:", img_freq[0],
      "and", img_freq[1], "(cycles per pixel).")
cutoff = sum(img_freq) / 2
# the cutoff frequency is around 0.0364 (cycles per pixel)
print("half way image frequency is", cutoff, "(cycles per pixel).")


# Part (c)
# Hint: fft2, ifft2, fftshift, and ifftshift functions all take an |axes|
# argument to specify the axes for the 2D DFT. e.g. fft2(arr, axes=(1, 2))
# Hint: Check out np.meshgrid.

img1_rgb = img1.astype(np.float64)/255
img2_rgb = img2.astype(np.float64)/255


def img_to_spectrum(rgb_image):
    """
    This function converts the input image's each channel
    of RGB to centralized spectrum
    :param rgb_image: an image in RGB
    :return: Each channel's corresponding spectrum
    """
    central_spec = np.zeros_like(rgb_image, np.complex_)
    for col in range(3):
        channel = rgb_image[:, :, col]
        # 2D Fast Fourier Transform
        img_spec = fft2(channel)
        # shift the spectrum
        central_spec[:, :, col] = fftshift(img_spec)
    return central_spec


# convert the images into spectrum by channel using FFT
img1_spec = img_to_spectrum(img1_rgb)
img2_spec = img_to_spectrum(img2_rgb)

#### Change these to the correct values for the high- and low-pass filters
hpf = np.zeros_like(img1[:, :, 0])
lpf = np.zeros_like(img1[:, :, 0])

nyquist_freq = 0.5  # largest frequency in the spectrum
cutoff_radius = cutoff / nyquist_freq * (W/2)   # set the cutoff freq
print(cutoff_radius)   # pixels from the center of fft image (cutoff radius)
# make the high-pass and low-pass filters
for i in range(W):
    for j in range(W):
        dist = np.sqrt((i-W/2)**2 + (j-W/2)**2)
        if dist <= cutoff_radius:
            lpf[i, j] = 1
        else:
            lpf[i, j] = 0
plt.imshow(np.abs(lpf), 'gray'); plt.show()
hpf = 1 - lpf
plt.imshow(np.abs(hpf), 'gray'); plt.show()

#### Apply the filters to create the hybrid image
hybrid_img = np.zeros_like(img1_rgb)

## Method 1: For each image each channel, convert RGB image to spectrum,
## apply the filters, convert each spectrum back to RGB image, then add
## the two images by each RGB channel.

def spectrum_to_img(img_spec, pass_filter):
    """
    This function takes an input image's three spectrum
    for each channel, apply the filter and shift back
    to each channel
    :param img_spec: spectrum for an image RGB channels
    :param pass_filter: high- or low-pass filter
    :return: an image in RBG with three channels
    """
    filtered_img = np.zeros_like(img_spec, np.complex_)
    for col in range(3):
        channel = img_spec[:, :, col]
        channel = np.multiply(pass_filter, channel)
        # 2D Fast Fourier Transform
        img_ishift = ifftshift(channel)
        # shift the spectrum
        filtered_img[:, :, col] = ifft2(img_ishift)
    return filtered_img.real


# Filter each image, transform spectrum back to RGB image
low_pass_img = spectrum_to_img(img1_spec, lpf)
high_pass_img = spectrum_to_img(img2_spec, hpf)

# plt.imshow(low_pass_img.clip(0, 1)); plt.show()
# plt.imshow(high_pass_img.clip(0, 1)); plt.show()

# Merge (add) the two filtered images by channels
for col in range(3):
    hybrid_img[:, :, col] = 255 * (low_pass_img[:, :, col] + high_pass_img[:, :, col])

plt.imshow(hybrid_img.clip(0,255).astype(np.uint8)); plt.show()

#=====================================================================

## Method 2: For each image each channel, convert RGB image to spectrum,
## add the two images' spectrum by each RGB channel, apply the filters
## to each channel's spectrum, then convert each spectrum back to RGB image.


# def spectrum_filter(img_spec, pass_filter):
#     """
#         This function takes an input image's three spectrum
#         for each channel, apply the filter to each channel
#         :param img_spec: spectrum for an image RGB channels
#         :param pass_filter: high- or low-pass filter
#         :return: an image with three filtered spectrum
#     """
#     filtered_spec = np.zeros_like(img_spec, np.complex_)
#     for col in range(3):
#         channel = img_spec[:, :, col]
#         filtered_spec[:, :, col] = np.multiply(pass_filter, channel)
#     return filtered_spec
#
# # apply the filter to the targeted spectrum
# low_pass_spec = spectrum_filter(img1_spec, lpf)
# high_pass_spec = spectrum_filter(img2_spec, hpf)
#
# hybrid_spec = np.zeros_like(img1_rgb, np.complex_)
#
# # Merge (add) the spectrum of two images by each channel
# for col in range(3):
#     hybrid_spec[:, :, col] = low_pass_spec[:, :, col] + high_pass_spec[:, :, col]
#
#
# def spectrum_to_img(img_spec):
#     """
#         This function takes an input image's three spectrum
#         for each channel and shift back to each channel
#         :param img_spec: spectrum for an image RGB channels
#         :return: an image in RBG with three channels
#     """
#     filtered_img = np.zeros_like(img_spec, np.complex_)
#     for col in range(3):
#         channel = img_spec[:, :, col]
#         # 2D Fast Fourier Transform
#         img_ishift = ifftshift(channel)
#         # shift the spectrum
#         filtered_img[:, :, col] = ifft2(img_ishift)
#     return filtered_img.real
#
# # convert the merged spectrum into RGB image
# hybrid_img = spectrum_to_img(hybrid_spec) * 255

#=====================================================================

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
axs[0,0].imshow(img2)
axs[0,0].axis('off')
axs[0,1].imshow(hpf, cmap='gray')
axs[0,1].set_title("High-pass filter")
axs[1,0].imshow(img1)
axs[1,0].axis('off')
axs[1,1].imshow(lpf, cmap='gray')
axs[1,1].set_title("Low-pass filter")
plt.savefig("hpf_lpf.png", bbox_inches='tight')
io.imsave("hybrid_image.png", np.clip(hybrid_img, a_min=0, a_max=255.))
# This line discard the Lossy conversion warning by
# adding .astype(np.uint8)), but thanks to io.imsave,
# it does not matter if not
# io.imsave("hybrid_image.png", np.clip(hybrid_img, a_min=0, a_max=255.).astype(np.uint8))