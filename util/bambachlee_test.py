import unittest
from .bambachlee import *
import cv2
import numpy as np

class TestBambachLee(unittest.TestCase):

    def test_peaks(self):
        frame = np.zeros((100,100,3))
        frame[4, np.arange(10, 15)] = (255, 0, 0)
        frame[5, np.arange(10, 15)] = (255, 0, 0)
        frame[6, np.arange(10, 15)] = (255, 0, 0)
        frame[4, np.arange(20, 25)] = (255, 0, 0)
        frame[5, np.arange(20, 25)] = (255, 0, 0)
        frame[6, np.arange(20, 25)] = (255, 0, 0)
        target_color = (255, 0, 0)

        coords = find_peaks(frame, target_color, [4,6], 2, threshold=0.01, step_size=1)
        self.assertTrue(np.alltrue(coords == [17]))

if __name__ == '__main__':
    unittest.main()
