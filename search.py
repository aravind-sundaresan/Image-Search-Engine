__author__ = "whackadoodle"
 
 
from numpy import *
import sys
import numpy as np
import cv2
import os
import cPickle as pickle
from getfeatures_ser import get_features,histogram
 
root_path='./dataset/'
 
limit=10
 
f=pickle.load(open('dict_index.pkl','rb'))
 
 
def chi2_distance(histA, histB, eps = 1e-10):
	# compute the chi-squared distance
	d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(histA, histB)])
 
	# return the chi-squared distance
	return d
 
def get_distance(query_features):
	distances={}
	for item in f.keys():
		features=f[item]
		d=chi2_distance(features, query_features)
		distances[item]=d
 
	sort_distances = sorted([(v, k) for (k, v) in distances.items()])
	return sort_distances[:4]
 
 
 
#query_patih='./queries/'
query_img_name= sys.argv[1]
query=cv2.imread(root_path+query_img_name)
query_features=get_features(query)
results=get_distance(query_features)
 
print "query image : ",sys.argv[1]
print '\n'
 
for (score, resultID) in results:
	print resultID ,' : ',score
	result = cv2.imread(root_path + resultID)
	#cv2.imwrite('./search_result/'+resultID,result)
	cv2.namedWindow("Display window",cv2.WINDOW_NORMAL)
	cv2.imshow("Display window",result)
	cv2.waitKey(0)
	#cv2.destroyAllWindows()
 
