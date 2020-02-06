import cv2
import numpy as np
import histogram_.plot_histogram as ph

# Constrast Limited Adaptive Histogram Equalization = CLAHE
# Input: an image
# Output: Adaptive histogram equalised image
def histogramEqualization_CLAHE(image):
    # creating the clahe object
    clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8))
    clahe_image = clahe.apply(image)
    result = np.hstack((image, clahe_image))
    cv2.imwrite("result.jpg", result)

    # analysing the before and after histograms
#    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
#    cdf = hist.cumsum()
#    cdf_n = cdf*hist.max()/cdf.max()
#    ph.plot_histogram(cdf_n, image)
    
#    hist, bins = np.histogram(clahe_image.flatten(), 256, [0, 256])
#    cdf = hist.cumsum()
#    cdf_n = cdf*hist.max()/cdf.max()
#    ph.plot_histogram(cdf_n, clahe_image)

    return clahe_image

# Global histogram equlization using opencv inbuilt functions
# Input: a grayscale image
# Output: a histogram equalized grayscale image
def histogramEqualization_opencv(image):
    equalized_image = cv2.equalizeHist(image)
    result = np.hstack((image, equalized_image))
    cv2.imwrite("result.jpg", result)
    return equalized_image


# Global histogram equalization using numpy library
# Input: an image
# Output: a histogram equalized image
def histogramEqualization_numpy(image):
    # getting the histogram and bins
    # 256 bins in the range [0, 256]
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    # getting the normalized cdf
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
#    ph.plot_histogram(cdf_normalized, image)
    # masking the zero sized bins: these bins are not involved in any further computation
    cdf_m = np.ma.masked_equal(cdf, 0)
    # lookup table: input is intensity and output is new intensity
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max() - cdf_m.min())
    lookUpTable = np.ma.filled(cdf_m, 0).astype('uint8')
    # transforming the image to histogram_equalized image
    transformedImage = lookUpTable[image]
    # getting the required data for plotting the new histogram
#    hist, bins = np.histogram(transformedImage.flatten(), 256, [0, 256])
#    cdf = hist.cumsum()
#    cdf_normalized = cdf * hist.max() / cdf.max()
#    ph.plot_histogram(cdf_normalized, transformedImage)

    result = np.hstack((image, transformedImage))
    cv2.imwrite('res.jpg', result)
    
    cv2.destroyAllWindows()
    return transformedImage
