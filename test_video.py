#!/bin/env/python
import os
import collections
import numpy as np
import cv2
import util

# Constant parameters used in Aruco methods
ARUCO_DICT = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

# prep path
SCRIPT_NAME = os.path.realpath(__file__)
SCRIPT_PATH = os.path.dirname(SCRIPT_NAME)

# load camera mats
FS = cv2.FileStorage(SCRIPT_PATH + "/calibration.xml", cv2.FILE_STORAGE_READ)
MTX = FS.getNode("camera_matrix").mat()
DIST = FS.getNode("camera_dist").mat()
FS.release()

# globals
WINDOW_TITLE = 'test video'


def detect_bars(img, mask):
    """detects bars using houghlines. this function draws horizontal lines on the provided mask"""
    edges = cv2.Canny(img, 50, 200, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 2, np.pi/180, 300,
                            minLineLength=220, maxLineGap=50)
    if lines is None:
        return

    for line in lines:
        x1, y1, x2, y2 = line[0]
        # only lines on the horizontal axis
        # inside a certain threshold
        if abs(y2-y1) < 20:
            cv2.line(mask, (x1, y1), (x2, y2), (255), 10)


def draw_rod_line(frame, rod_candidates):
    """draws all rod candidates line on the frame"""
    (_, w) = frame.shape[0:2]

    for _, (y_0, y_1) in enumerate(rod_candidates):
        y_mid = y_0 + (y_1-y_0)
        cv2.line(frame, (0, y_mid), (w, y_mid), (0, 0, 255), 1)


def main():
    cv2.namedWindow(WINDOW_TITLE)
    cap = cv2.VideoCapture('test.mp4')

    prev_markers = []
    paused = False
    calibration_frames = 50
    calibration_mask = None
    rod_candidates = []

    while(cap.isOpened()):
        # if not paused:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = util.img.resize_image_factor(frame, 0.5)
        frame = util.img.rotate_image(frame, -90)

        #
        # we use aruco markers to detect the edges of the foosball table
        # we warp the image so that the we only see the playing field
        #
        corners, ids, _ = cv2.aruco.detectMarkers(
            frame, ARUCO_DICT, cameraMatrix=MTX, distCoeff=DIST)

        tmp = {}
        for i, marker in enumerate(ids):
            tmp[marker[0]] = corners[i]

        markers = collections.OrderedDict(sorted(tmp.items()))

        if len(corners) != 4:
            # re-use prev markers
            if len(prev_markers) == 4:
                markers = prev_markers
            else:
                print("missing corners. skipping")
                continue

        frame = util.img.warp_image(frame, markers)

        # calibration mode: use HoughLinesP to find the bars
        #
        # * detect rods
        # * detect figures
        if calibration_frames > 0:
            if calibration_mask is None:
                calibration_mask = np.zeros(frame.shape[:2], np.uint8)

            detect_bars(frame, calibration_mask)
            cv2.imshow(WINDOW_TITLE, calibration_mask)
            key = cv2.waitKey(10)
            if key == 27:
                break

            calibration_frames = calibration_frames-1
            if calibration_frames >= 1:
                continue

            # the last calibration iteration
            # will calculate the rods

            # 3d (x, y, int) -> (y, int)
            dst = np.max(calibration_mask, 1)

            # helper fn
            # get y positions of rod
            #
            last_state = False
            candidate = None
            for i, row in enumerate(dst):
                # start of rod
                if last_state is False and row == 255:
                    candidate = [i, 0]
                    last_state = True
                    continue

                # end of rod
                if (last_state is True and row != 255) or (last_state is True and i == len(dst)):
                    delta = i - candidate[0]
                    mid = candidate[0] + (delta//2)
                    candidate[0] = mid-1
                    candidate[1] = mid+1
                    rod_candidates.append(candidate)
                    last_state = False
                    continue

        prev_markers = markers

        # these values must be calibrated for each
        # color scheme and lighting situation
        # beware: they are HSV
        green = (110, 68, 44)
        black = (255, 255, 255)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        for i, candidate in enumerate(rod_candidates):
            if i in (2, 4, 6, 7):
                pts = util.bambachlee.find_peaks(
                    hsv, black, candidate, sigma=20, threshold=0.5)
            else:
                pts = util.bambachlee.find_peaks(hsv, green, candidate, threshold=0.7)
            for p in pts:
                cv2.line(frame, (p, candidate[0]), (p, candidate[1]), (255, 0, 0), 5)

        draw_rod_line(frame, rod_candidates)

        cv2.imshow(WINDOW_TITLE, frame)
        key = cv2.waitKey(1)
        if key == 32:  # space
            paused = not paused
        if key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
