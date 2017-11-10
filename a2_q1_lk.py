'''
READ ME FIRST :
    PLEASE COMMENT/UNCOMMENT THE RESPECTIVE IMAGES BEFORE RUNNING
    FOR A VALID REASON THAT I DO NOT WISH TO PICTURE BOMB THE SCREEN
    COMMENT/UNCOMMENT AT THE VERY BOTTOM OF THE CODE
'''

import cv2 as cv
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

def load_images(files):
    file1, file2 = files[0], files[1]
    image1 = cv.imread(file1, cv.IMREAD_GRAYSCALE)
    pixels1 = image1.shape[0] * image1.shape[1]
    cv.imshow('Grayscale Image ~ ' + file1, image1)
    cv.waitKey(1)
    image2 = cv.imread(file2, cv.IMREAD_GRAYSCALE)
    pixels2 = image2.shape[0] * image2.shape[1]
    cv.imshow('Grayscale Image ~ ' + file2, image2)
    cv.waitKey(1)
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
    plt.figure('Corners for Image ~ ' + file2)
    plt.imshow(image2, cmap='gray')
    plt.plot(cornersX, cornersY, 'r.')
    return None

def plot_corners_looped(params):
    file2, image2 = params[0], params[1]
    cornersX, cornersY = params[2], params[3]
    loop = params[4]
    plt.figure('Corners after Gaussian Downsampling for Image ~ ' + file2 + ' ~ Loop - ' + str(loop))
    plt.imshow(image2, cmap='gray')
    plt.plot(cornersX, cornersY, 'r.')
    return None

def scale_down_imgs(params):
    image1, image2, scale = params[0], params[1], params[2]
    image1_resized = cv.resize(image1, (int(image1.shape[1] / scale), int(image1.shape[0] / scale)), interpolation = cv.INTER_CUBIC)
    image2_resized = cv.resize(image2, (int(image2.shape[1] / scale), int(image2.shape[0] / scale)), interpolation = cv.INTER_CUBIC)
    return image1_resized, image2_resized

def lucas_kanade_field(params):
    image1, image2 = params[0], params[1]
    window_size = params[2]
    # Calculating Ix Iy and It for each point
    mx = np.array([[-1, 1],[-1, 1]]).reshape(2, 2)
    my = np.array([[-1, -1],[1, 1]]).reshape(2, 2)
    # Partial Derivative on X
    Ix_m = ndi.convolve(image1, mx)
    # Partial Derivative on Y
    Iy_m = ndi.convolve(image1, my)
    # Partial Derivative on t
    It_m = ndi.convolve(image1, np.ones([2, 2])) + ndi.convolve(image2, -np.ones([2, 2]))
    u = np.zeros(image1.shape)
    v = np.zeros(image2.shape)
    # Within specified Window_size^2
    for i in range(int(window_size + 1), int(Ix_m.shape[0] - window_size + 1)):
        for j in range(int(window_size + 1), int(Ix_m.shape[1] - window_size + 1)):
            Ix = Ix_m[int(i - window_size):int(i + window_size + 1), int(j - window_size):int(j + window_size + 1)]
            Iy = Iy_m[int(i - window_size):int(i + window_size + 1), int(j - window_size):int(j + window_size + 1)]
            It = It_m[int(i - window_size):int(i + window_size + 1), int(j - window_size):int(j + window_size + 1)]
            # IX, IY and B are column vectors of Ix, Iy and It stacked column-wise
            IX = np.array(Ix.T.ravel())
            IY = np.array(Iy.T.ravel())
            B = np.array(-It.T.ravel())
            A = np.vstack([IX, IY]).T
            nu = np.matmul(np.linalg.pinv(A), B)
            u[i, j] = nu[0]
            v[i, j] = nu[1]
    # Downsize/Decimate u and v
    u_deci = u[:, ::10][::10, :]
    v_deci = v[:, ::10][::10, :]
    # get coords of u and v in original frame
    m, n = image1.shape
    x = np.array([i for i in range(0, n)])
    y = np.array([j for j in range(0, m)])
    X, Y = np.meshgrid(x, y)
    X_deci = X[:, ::10][::10, :]
    Y_deci = Y[:, ::10][::10, :]
    return X_deci, Y_deci, u_deci, v_deci

def draw_optfl_fld(params):
    file2, image2 = params[0], params[1]
    X, Y, u, v = params[2], params[3], params[4], params[5]
    plt.figure('Optical Flow Field using Lucas-Kanade Method ~ ' + file2)
    plt.imshow(image2, cmap='gray')
    plt.quiver(X, Y, u, v, units='width', color='r')
    return None

def draw_optfl_fld_pyr(params):
    file2, image2 = params[0], params[1]
    X, Y, u, v = params[2], params[3], params[4], params[5]
    plt.figure('Optical Flow Field after Gaussian Downsampling using Lucas-Kanade Method ~ ' + file2)
    plt.imshow(image2, cmap='gray')
    plt.quiver(X, Y, u, v, units='width', color='r')
    return None

def draw_optfl_fld_pyr_looped(params):
    file2, image2 = params[0], params[1]
    X, Y, u, v = params[2], params[3], params[4], params[5]
    loop = params[6]
    plt.figure('Optical Flow Field after Gaussian Downsampling using Lucas-Kanade Method ~ ' + file2 + ' ~ Loop - ' + str(loop))
    plt.imshow(image2, cmap='gray')
    plt.quiver(X, Y, u, v, units='width', color='r')
    return None

def lucas_kanade_vectors(params):
    image1, image2 = params[0], params[1]
    window_size = params[2]
    cornersX, cornersY = params[3], params[4]
    # Calculating Ix Iy and It for each point
    mx = np.array([[-1, 1],[-1, 1]]).reshape(2, 2)
    my = np.array([[-1, -1],[1, 1]]).reshape(2, 2)
    # Partial Derivative on X
    Ix_m = ndi.convolve(image1, mx)
    # Partial Derivative on Y
    Iy_m = ndi.convolve(image1, my)
    # Partial Derivative on t
    It_m = ndi.convolve(image1, np.ones([2, 2])) + ndi.convolve(image2, -np.ones([2, 2]))
    u = np.zeros(cornersX.shape)
    v = np.zeros(cornersY.shape)
    # Within specified Window_size^2
    for k in range(0, len(cornersX)):
        i = cornersY[k]
        j = cornersX[k]
        Ix = Ix_m[int(i - window_size):int(i + window_size + 1), int(j - window_size):int(j + window_size + 1)]
        Iy = Iy_m[int(i - window_size):int(i + window_size + 1), int(j - window_size):int(j + window_size + 1)]
        It = It_m[int(i - window_size):int(i + window_size + 1), int(j - window_size):int(j + window_size + 1)]
        # IX, IY and B are column vectors of Ix, Iy and It stacked column-wise
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
    plt.figure('Optical Flow Vectors using Lucas-Kanade Method ~ ' + file2)
    plt.imshow(image2, cmap='gray')
    plt.quiver(cornersX, cornersY, u, v, units='width', color='r')
    return None

def draw_optfl_vecs_pyr(params):
    file2, image2 = params[0], params[1]
    cornersX, cornersY = params[2], params[3]
    u, v = params[4], params[5]
    plt.figure('Optical Flow Vectors after Gaussian Downsampling using Lucas-Kanade Method ~ ' + file2)
    plt.imshow(image2, cmap='gray')
    plt.quiver(cornersX, cornersY, u, v, units='width', color='r')
    return None

def draw_optfl_vecs_pyr_looped(params):
    file2, image2 = params[0], params[1]
    cornersX, cornersY = params[2], params[3]
    u, v = params[4], params[5]
    loop = params[6]
    plt.figure('Optical Flow Vectors after Gaussian Downsampling using Lucas-Kanade Method ~ ' + file2 + ' ~ Loop - ' + str(loop))
    plt.imshow(image2, cmap='gray')
    plt.quiver(cornersX, cornersY, u, v, units='width', color='r')
    return None

def unit_pyr_down(params):
    image1 = params[0]
    file2, image2 = params[1], params[2]
    window_size, scale = params[3], params[4]
    max_corners, quality, min_distance = params[5], params[6], params[7]
    sigma = params[8]
    algo_correct = params[9]
    # Gaussian Blur
    image1 = cv.GaussianBlur(image1, (5, 5), sigma)
    image2 = cv.GaussianBlur(image2, (5, 5), sigma)
    # Downsample Images for Field
    image1_resized, image2_resized = scale_down_imgs([image1, image2, scale])
    # Downsample and Find Corners for Vectors
    file2, image2, cornersX, cornersY, window_size = find_corners([file2, image2, window_size, scale, max_corners, quality, min_distance])
    plot_corners([file2, image2, cornersX, cornersY])
    # Lucas-Kanade Field
    x_fld, y_fld, u_fld, v_fld = lucas_kanade_field([image1_resized, image2_resized, window_size])
    draw_optfl_fld_pyr([file2, image2, x_fld*scale, y_fld*scale, u_fld*scale, v_fld*scale*algo_correct])
    # Lucas-Kanade Vectors
    u_vec, v_vec = lucas_kanade_vectors([image1, image2, window_size, cornersX, cornersY])
    draw_optfl_vecs_pyr([file2, image2, cornersX, cornersY, u_vec, v_vec*algo_correct])
    # Return Downsampled Images
    return image1_resized, image2_resized

def looped_pyr_down(params):
    image1 = params[0]
    file2, image2 = params[1], params[2]
    window_size, scale = params[3], params[4]
    max_corners, quality, min_distance = params[5], params[6], params[7]
    sigma = params[8]
    loop = params[9]
    algo_correct = params[10]
    # Gaussian Blur
    image1 = cv.GaussianBlur(image1, (5, 5), sigma)
    image2 = cv.GaussianBlur(image2, (5, 5), sigma)
    # Downsample Images for Field
    image1_resized, image2_resized = scale_down_imgs([image1, image2, scale])
    # Downsample and Find Corners for Vectors
    file2, image2, cornersX, cornersY, window_size = find_corners([file2, image2, window_size, scale, max_corners, quality, min_distance])
    plot_corners_looped([file2, image2, cornersX, cornersY, loop])
    # Lucas-Kanade Field
    x_fld, y_fld, u_fld, v_fld = lucas_kanade_field([image1_resized, image2_resized, window_size])
    draw_optfl_fld_pyr_looped([file2, image2, x_fld*scale, y_fld*scale, u_fld*scale, v_fld*scale*algo_correct, loop])
    # Lucas-Kanade Vectors
    u_vec, v_vec = lucas_kanade_vectors([image1, image2, window_size, cornersX, cornersY])
    draw_optfl_vecs_pyr_looped([file2, image2, cornersX, cornersY, u_vec, v_vec*algo_correct, loop])
    # Return Downsampled Images for Looping
    return image1_resized, image2_resized


def do_lucas_kanade(params):
    # Implementation on Images
    file1, file2 = params[0], params[1]
    window, sigma = params[2], params[3]
    algo_correct = params[4]
    file1, file2, image1, image2, pixels1, pixels2 = load_images([file1, file2])
    file2, image2, cornersX, cornersY, window_size = find_corners([file2, image2, window, 1, pixels2, 0.01, 10])
    plot_corners([file2, image2, cornersX, cornersY])
    x_fld, y_fld, u_fld, v_fld = lucas_kanade_field([image1, image2, window_size])
    draw_optfl_fld([file2, image2, x_fld, y_fld, u_fld, v_fld*algo_correct])
    u_vec, v_vec = lucas_kanade_vectors([image1, image2, window_size, cornersX, cornersY])
    draw_optfl_vecs([file2, image2, cornersX, cornersY, u_vec, v_vec*algo_correct])
    #image1, image2 = unit_pyr_down([image1, file2, image2, window_size, 2, pixels2, 0.01, 10, 1.0, algo_correct])
    # Subsample the Image 4 times for Pyramid
    for loop in range(1, 5):
        image1, image2 = looped_pyr_down([image1, file2, image2, window_size, 2, int(image2.shape[0] * image2.shape[1]), 0.01, 10, sigma, loop, algo_correct])
    return None
    

'''PLEASE COMMENT/UNCOMMENT THE RESPECTIVE IMAGES BEFORE RUNNING HERE'''
'''Function : do_lucas_kanade([filename1, filename2, window_size, sigma_blur, algorithm_correction])'''

'''ALGORITHM ERROR CORRECTION EXPLAINED IN THE ATTACHED PDF DOCUMENT'''

'''WITHOUT ALGORITHM ERROR CORRECTION'''
'''COMMENT BELOW TWO LINES BEFORE RUNNING THIS'''
# Basketball
#do_lucas_kanade(['basketball1.png', 'basketball2.png', 20, 0.5, 1])
# Grove
#do_lucas_kanade(['grove1.png', 'grove2.png', 20, 0.5, 1])

'''WITH ALGORITHM ERROR CORRECTION'''
'''COMMENT ABOVE TWO LINES BEFORE RUNNING THIS'''
# Basketball
do_lucas_kanade(['basketball1.png', 'basketball2.png', 20, 0.5, 1])
# Grove
#do_lucas_kanade(['grove1.png', 'grove2.png', 20, 0.5, -1])


# End of File