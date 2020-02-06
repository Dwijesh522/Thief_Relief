import cv2
import numpy as np
from matplotlib import pyplot as plt
import histogram_.histogram_equalization as he
import clustering.kmeans_clustering as kc
import io_.read_image as ri

if __name__ == "__main__":
    # <read image>
    image = ri.readImage_gray()
    transformed_image = he.histogramEqualization_CLAHE(image)
    clustered_image = kc.kmeans_clustering(image, 50)
    cv2.imwrite("result.jpg", clustered_image)
