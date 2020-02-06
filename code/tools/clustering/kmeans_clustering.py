import numpy as np
import cv2

# Input: an image and cluster size
# Output: clusterized image
def kmeans_clustering(image, cluster_size):
    # creating column of each feature. Here R column, G column, and B column for an rgb image
    feature_columns = image.reshape((-1, 3))
    # typecasting to numpy float 32
    feature_columns = np.float32(feature_columns)
    # criteria: type, iteration threshold, epsilon
    ITERATION_THRESHOLD = 10
    EPSILON = 1
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, ITERATION_THRESHOLD, EPSILON)
    # minimized squared distance, labels to each observation, set of centers
    ret, label, center = cv2.kmeans(feature_columns, cluster_size, None, criteria, ITERATION_THRESHOLD, cv2.KMEANS_RANDOM_CENTERS)
    # typecasting to uint8 and getting back the image type
    center = np.uint8(center)
    # replacing each image pixel with its nearest mean
    clustered_image = center[label.flatten()]
    clustered_image = clustered_image.reshape((image.shape))

    return clustered_image
