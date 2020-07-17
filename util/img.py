import cv2
import numpy as np

def rotate_image(img, angle):
    (height, width) = img.shape[:2]
    (cX, cY) = (width // 2, height // 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((height * sin) + (width * cos))
    nH = int((height * cos) + (width * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(img, M, (nW, nH))

def resize_image_factor(img, factor):
    width = int(img.shape[1] * factor)
    height = int(img.shape[0] * factor)
    dsize = (width, height)
    return cv2.resize(img, dsize)

def resize_image_width(img, width):
    aspect_ratio = img.shape[1] / img.shape[0]
    width = int(width)
    height = int(width / aspect_ratio)
    return cv2.resize(img, (width, height))

def warp_image(img, markers):
    (height, width) = img.shape[:2]
    src_points = np.array([])
    topX = 0
    topY = 0
    minX = float('inf')
    minY = float('inf')
    # order is important

    for _, corner in markers.items():
        src_points = np.append(src_points, corner[0][0])

    src_points = src_points.reshape(4, 1, -1)
    pts_dst = np.array(
        [width, height, 0, height, 0, 0, width, 0]).reshape(4, 1, -1)

    # src points are not in correct order
    H = cv2.findHomography(src_points, pts_dst)
    return cv2.warpPerspective(img, H[0], (width, height))
