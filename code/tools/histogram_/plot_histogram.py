# Input: Histogram cdf, image
# Output: plot of histogram, cdf of histogram

from matplotlib import pyplot as plt

def plot_histogram(cdf, image):
    plt.plot(cdf, color = 'b')
    plt.hist(image.flatten(), 256, [0, 256], color = 'r')
    plt.xlim([0, 256])
    plt.legend(('CDF', 'Histogram'), loc = 'upper left')
    plt.show()
