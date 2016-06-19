#!/bin/python
# ref
# http://www.pyimagesearch.com/2014/05/26/opencv-python-k-means-color-clustering/
# kd-tree for nearest vectors searching 
# http://stackoverflow.com/questions/32446703/find-closest-vector-from-a-list-of-vectors-python
from sklearn.cluster import KMeans
import cv2
import numpy as np
def centroid_histogram(clt):
	numLabels = np.arange(0, len(np.unique(clt.labels_)) +1)
	(hist, _) = np.histogram(clt.labels_, bins = numLabels)
	
	# normalize the histogram
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

def separate_colors(im, centroids):
 	# using spatial kd-tree for searching
	from scipy import spatial
	# black as 0th color
	centroids = np.append([[0,0,0]], centroids, axis=0)
	tree = spatial.KDTree(centroids)
	# test data: pixels
	pixels = im.reshape((im.shape[0] * im.shape[1], 3))
	# labels
 	labels = []
	for p in pixels:
		dist, idx = tree.query(p)
		labels.append(idx)
	labels = np.array(labels).reshape((im.shape[0], im.shape[1]))
	# np.unique(labels)
	# np.savetxt('labels',labels, fmt = '%5d')
	for l in range(0, len(centroids)):
		lim = np.not_equal(labels,l)
		lim = lim.astype("uint8")
		# lim = 255*lim.astype("uint8")
		masked = cv2.bitwise_and(im, im, mask=lim)
		cv2.imwrite(str(l)+".jpg",masked)
		
def cluster_colors(im):
	for i in xrange(6):
		im = cv2.pyrDown(im)
	# im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
	# reshape im to a vector of pixels
	im_ = im.reshape((im.shape[0] * im.shape[1], 3))
	pixels = []
	for p in im_:
		if list(p)!=[0,0,0]:
			pixels.append(p)
	pixels = np.array(pixels)	

	# cluster the pixel intensities, n_clusters
	clt = KMeans(n_clusters = 15)
	clt.fit(pixels)

	# build histogram
	hist = centroid_histogram(clt)
	bar = plot_colors(hist, clt.cluster_centers_)
	cv2.imwrite('bar.jpg', bar)
	cv2.imwrite('im.jpg', im)

	# separate image by labels
	separate_colors(im, clt.cluster_centers_)


if __name__=="__main__":
	im = cv2.imread('data/1.jpg')
	cluster_colors(im)
