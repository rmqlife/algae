#!/bin/python
# ref
# http://www.pyimagesearch.com/2014/05/26/opencv-python-k-means-color-clustering/
from sklearn.cluster import KMeans
import cv2
import numpy as np
def centroid_histogram(clt):
	numLabels = np.arange(0, len(np.unique(clt.labels_)) +1)
	(hist, _) = np.histogram(clt.labels_, bins = numLabels)
	
	hist = hist.astype("float")
	hist /= hist.sum()
	return hist

def plot_colors(hist, centroids):
	bar = np.zeros((50,300,3), dtype = "uint8")
	startX = 0;
	
	for (percent, color) in zip(hist, centroids):
		endX = startX + (percent * 300)
		cv2.rectangle(bar, (int(startX),0), (int(endX), 50),\
		 color.astype("uint8").tolist(), -1)
		startX = endX
	return bar

def is_black(s): 
	return np.array_equal(s, np.array([0,0,0]))			


def clusterColor(im):
	
	for i in xrange(2):
		im = cv2.pyrDown(im)
		cv2.imshow('im', im)

	# im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
	# reshape im to a vector of pixels
	im = im.reshape((im.shape[0] * im.shape[1], 3))
	pixels = []
	for p in im:
		if list(p)!=[0,0,0]:
			pixels.append(p)
	im = np.array(pixels)	

	# cluster the pixel intensities, n_clusters
	clt = KMeans(n_clusters = 10)
	clt.fit(im)

	# build histogram
	hist = centroid_histogram(clt)
	bar = plot_colors(hist, clt.cluster_centers_)

	cv2.imshow('bar', bar)
	cv2.waitKey(0)

if __name__=="__main__":
	im = cv2.imread('data/1.jpg')
	clusterColor(im)
