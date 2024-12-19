'''
Capture - various screen capture routines

'''

import numpy as np
import cv2
import mss

sct = mss.mss()

def capture_gray(area=(0, 0, 640, 480), out_size=(128, 96)):
	pixels = sct.grab({'top': area[0], 'left': area[1], 'width': area[2], 'height': area[3]})
	pixels = np.asarray(pixels)
	pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
	pixels = cv2.resize(pixels, out_size, interpolation=cv2.INTER_AREA)
	return pixels

def main():
	while True:
		pixels = capture_gray()

		cv2.imshow('w', pixels)
		cv2.waitKey(1)

if __name__ == '__main__':
	main()