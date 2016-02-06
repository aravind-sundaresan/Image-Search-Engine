__author__ = "whackadoodle"
 
from numpy import *
import sys
import numpy as np
import cv2
import os
import csv
import cPickle as pickle
 
 
 
 
def histogram(image, mask):
	bins=(8,12,3)
	hist = cv2.calcHist([image], [0, 1, 2], mask, bins,[0, 180, 0, 256, 0, 256])
	hist = cv2.normalize(hist).flatten()
	return hist
 
def get_features(image_original):
	image=cv2.cvtColor(image_original, cv2.COLOR_BGR2HSV)
 
	(h, w) = image.shape[:2]
 
	(cX, cY) = (int(w * 0.5), int(h * 0.5))
 
	segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h),(0, cX, cY, h)]
 
	(axesX, axesY) = (int(w * 0.75) / 2, int(h * 0.75) / 2)
	ellipMask = np.zeros(image.shape[:2], dtype = "uint8")
	cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)
 
 
	features=[]
 
	# loop over the segments
	for (startX, endX, startY, endY) in segments:
		# construct a mask for each corner of the image, subtracting
		# the elliptical center from it
		cornerMask = np.zeros(image.shape[:2], dtype = "uint8")
		cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
		cornerMask = cv2.subtract(cornerMask, ellipMask)
 
		# extract a color histogram from the image, then update the
		# feature vector
		hist = histogram(image, cornerMask)
		features.extend(hist)
 
	# extract a color histogram from the elliptical region and
	# update the feature vector
	hist = histogram(image, ellipMask)
	features.extend(hist)
 
	# return the feature vector
	return features
 
 
 
 
if __name__ == '__main__':
	root_path='./dataset/'
	index={}
	temp=0
	for items in os.listdir(root_path):
		if items[-4:]=='.png':
			temp+=1
			image=cv2.imread(root_path+items)
			feature=get_features(image)
			index[items]=feature
			print temp
 
 
 
	with open('dict_index.pkl','wb') as f_hsv:
		pickle.dump(index,f_hsv)
 
