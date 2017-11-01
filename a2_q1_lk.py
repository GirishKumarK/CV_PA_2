import cv2 as cv
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

def load_images(files):
    file1, file2 = files[0], files[1]
    image1 = cv.imread(file1, cv.IMREAD_GRAYSCALE)
    pixels1 = image1.shape[0] * image1.shape[1]
    cv.imshow('Grayscale Image : ' + file1, image1)
    image2 = cv.imread(file2, cv.IMREAD_GRAYSCALE)
    pixels2 = image2.shape[0] * image2.shape[1]
    cv.imshow('Grayscale Image : ' + file2, image2)
    return file1, file2, image1, image2, pixels1, pixels2

def find_corners(params):
    file2 = params[0]
    image2 = params[1]
    window_size = np.round(params[2] / 2)
    scale = params[3]
    max_corners, quality, min_distance = params[4], params[5], params[6]
    # Resize Image
    image2_resized = cv.resize(image2, (int(image2.shape[1] / scale), int(image2.shape[0] / scale)), interpolation = cv.INTER_CUBIC)
    corners = cv.goodFeaturesToTrack(image2_resized, max_corners, quality, min_distance)
    corners = corners * scale
    cornersx = corners[:, :, 0].ravel()
    cornersy = corners[:, :, 1].ravel()
    # Discard Corners Near Image Margins
    new_corners = []
    for i in range(0, len(corners)):
        xi = cornersx[i]
        yi = cornersy[i]
        if ((xi - window_size) >= 1) and ((yi - window_size) >= 1) and ((xi + window_size) <= (image2.shape[1] - 1)) and ((yi + window_size) <= (image2.shape[0] - 1)):
            new_corners.append([xi, yi])
    new_corners = np.array(new_corners)
    cornersX = new_corners[:, 0].ravel()
    cornersY = new_corners[:, 1].ravel()
    return file2, image2, cornersX, cornersY, window_size

def plot_corners(params):
    file2, image2 = params[0], params[1]
    cornersX, cornersY = params[2], params[3]
    plt.figure('Corners for Image : ' + file2)
    plt.imshow(image2, cmap='gray')
    plt.plot(cornersX, cornersY, 'r.')
    return None

def lucas_kanade(params):
    image1, window_size = params[0], params[1]
    cornersX, cornersY = params[2], params[3]
    # Calculating Ix Iy and It for each point
    mx = np.array([[-1, 1],[-1, 1]]).reshape(2, 2)
    my = np.array([[-1, -1],[1, 1]]).reshape(2, 2)
    # Partial Derivative on X
    Ix_m = ndi.convolve(image1, mx)
    # Partial Derivative on Y
    Iy_m = ndi.convolve(image1, my)
    # Partial Derivative on t
    It_m = ndi.convolve(image1, mx) + ndi.convolve(image1, my)
    u = np.zeros(cornersX.shape)
    v = np.zeros(cornersY.shape)
    # Within specified Window_size^2
    for k in range(0, len(cornersX)):
        i = cornersY[k]
        j = cornersX[k]
        Ix = Ix_m[int(i - window_size):int(i + window_size + 1), int(j - window_size):int(j + window_size + 1)]
        Iy = Iy_m[int(i - window_size):int(i + window_size + 1), int(j - window_size):int(j + window_size + 1)]
        It = It_m[int(i - window_size):int(i + window_size + 1), int(j - window_size):int(j + window_size + 1)]
        IX = np.array(Ix.T.ravel())
        IY = np.array(Iy.T.ravel())
        B = np.array(-It.T.ravel())
        A = np.vstack([IX, IY]).T
        nu = np.matmul(np.linalg.pinv(A), B)
        u[k] = nu[0]
        v[k] = nu[1]
    return u, v

def draw_optfl_vecs(params):
    file2, image2 = params[0], params[1]
    cornersX, cornersY = params[2], params[3]
    u, v = params[4], params[5]
    plt.figure('Optical Flow Vectors using Lucas-Kanade Method : ' + file2)
    plt.imshow(image2, cmap='gray')
    plt.quiver(cornersX, cornersY, u, v, units='width', color='r')
    return None

def draw_optfl_vecs_pyr(params):
    file2, image2 = params[0], params[1]
    cornersX, cornersY = params[2], params[3]
    u, v = params[4], params[5]
    plt.figure('Optical Flow Vectors after Gaussian Downsampling using Lucas-Kanade Method : ' + file2)
    plt.imshow(image2, cmap='gray')
    plt.quiver(cornersX, cornersY, u, v, units='width', color='r')
    return None

def draw_optfl_vecs_pyr_looped(params):
    file2, image2 = params[0], params[1]
    cornersX, cornersY = params[2], params[3]
    u, v = params[4], params[5]
    loop = params[6]
    plt.figure('Optical Flow Vectors after Gaussian Downsampling using Lucas-Kanade Method : ' + file2 + ' ~ Loop : ' + str(loop))
    plt.imshow(image2, cmap='gray')
    plt.quiver(cornersX, cornersY, u, v, units='width', color='r')
    return None

def unit_pyr_down(params):
    image1 = params[0]
    file2, image2 = params[1], params[2]
    window_size, scale = params[3], params[4]
    max_corners, quality, min_distance = params[5], params[6], params[7]
    sigma = params[8]
    # Gaussian Blur
    image1 = cv.GaussianBlur(image1, (5, 5), sigma)
    image2 = cv.GaussianBlur(image2, (5, 5), sigma)
    # Down Sample and Find Corners
    file2, image2, cornersX, cornersY, window_size = find_corners([file2, image2, window_size, scale, max_corners, quality, min_distance])
    # Lucas Kanade
    u, v = lucas_kanade([image1, window_size, cornersX, cornersY])
    draw_optfl_vecs_pyr([file2, image2, cornersX, cornersY, u, v])
    return None

def looped_pyr_down(params):
    image1 = params[0]
    file2, image2 = params[1], params[2]
    window_size, scale = params[3], params[4]
    max_corners, quality, min_distance = params[5], params[6], params[7]
    sigma = params[8]
    loop = params[9]
    # Gaussian Blur
    image1 = cv.GaussianBlur(image1, (5, 5), sigma)
    image2 = cv.GaussianBlur(image2, (5, 5), sigma)
    # Down Sample and Find Corners
    file2, image2, cornersX, cornersY, window_size = find_corners([file2, image2, window_size, scale, max_corners, quality, min_distance])
    # Lucas Kanade
    u, v = lucas_kanade([image1, window_size, cornersX, cornersY])
    draw_optfl_vecs_pyr_looped([file2, image2, cornersX, cornersY, u, v, loop])
    return None


file1, file2, image1, image2, pixels1, pixels2 = load_images(['traffic1.png', 'traffic2.png'])
file2, image2, cornersX, cornersY, window_size = find_corners([file2, image2, 20, 1, pixels2, 0.01, 10])
plot_corners([file2, image2, cornersX, cornersY])
u, v = lucas_kanade([image1, window_size, cornersX, cornersY])
draw_optfl_vecs([file2, image2, cornersX, cornersY, u, v])
#unit_pyr_down([image1, file2, image2, window_size, 2, pixels2, 0.01, 10, 1.0])
for loop in range(1, 10):
    scale = loop**2
    looped_pyr_down([image1, file2, image2, window_size, scale, int(pixels2/2), 0.01, 10, 1.5, loop])

file1, file2, image1, image2, pixels1, pixels2 = load_images(['basketball1.png', 'basketball2.png'])
file2, image2, cornersX, cornersY, window_size = find_corners([file2, image2, 20, 1, pixels2, 0.01, 10])
plot_corners([file2, image2, cornersX, cornersY])
u, v = lucas_kanade([image1, window_size, cornersX, cornersY])
draw_optfl_vecs([file2, image2, cornersX, cornersY, u, v])
#unit_pyr_down([image1, file2, image2, window_size, 2, pixels2, 0.01, 10, 1.0])
for loop in range(1, 10):
    scale = loop**2
    looped_pyr_down([image1, file2, image2, window_size, scale, int(pixels2/2), 0.01, 10, 1.5, loop])

file1, file2, image1, image2, pixels1, pixels2 = load_images(['grove1.png', 'grove2.png'])
file2, image2, cornersX, cornersY, window_size = find_corners([file2, image2, 20, 1, pixels2, 0.01, 10])
plot_corners([file2, image2, cornersX, cornersY])
u, v = lucas_kanade([image1, window_size, cornersX, cornersY])
draw_optfl_vecs([file2, image2, cornersX, cornersY, u, v])
#unit_pyr_down([image1, file2, image2, window_size, 2, pixels2, 0.01, 10, 1.0])
for loop in range(1, 10):
    scale = loop**2
    looped_pyr_down([image1, file2, image2, window_size, scale, int(pixels2/2), 0.01, 10, 1.5, loop])


# End of File