# Input: path to an image
# Output: opencv image

import cv2
# return gray image
def readImage_gray():
    print("Default path: /home/dwijesh/Documents/.thief_relief/images/")
    path = input("Enter the path to an image: ")
    image = cv2.imread(path, 0)
    return image
# return rgb image
def readImage_rgb():
    print("Default path: /home/dwijesh/Documents/.thief_relief/images/")
    path = input("Enter the path to an image: ")
    image = cv2.imread(path)
    return image
