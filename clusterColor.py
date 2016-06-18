#!/bin/python
# ref
# http://www.pyimagesearch.com/2014/05/26/opencv-python-k-means-color-clustering/
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
	pixels = im.reshape((im.shape[0] * im.shape[1], 3))
 	# labels
 	labels = []
	from scipy.spatial import distance
	for p in pixels:
		if list(p)==[0,0,0]:
			labels.append(-1);
		else:
			dist = []
			for color in centroids:
				dist.append( distance.euclidean(p,color))
			labels.append(np.argmin(np.array(dist)))
	labels = np.array(labels).reshape((im.shape[0], im.shape[1]))
	# np.savetxt('labels',labels, fmt = '%5d')
	for l in range(0, len(centroids)):
		lim = np.not_equal(labels,l)
		lim = lim.astype("uint8")
		masked = cv2.bitwise_and(im, im, mask=lim)
		# lim = 255*lim.astype("uint8")
		# np.savetxt('labels',lim, fmt = '%5d')
		cv2.imwrite(str(l)+".jpg",masked)
	

				
def cluster_colors(im):
	for i in xrange(2):
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
