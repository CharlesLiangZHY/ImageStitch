import sys
import random
import cv2
import numpy as np

from image import Image
from image import resize
from image import visualizeMatch
from align import *


def stitch(img1, img2):
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            if img1[i,j,0] != 0:
                img2[i,j] = img1[i,j]
    return img2

if __name__ == "__main__":
    
    path = []
    if len(sys.argv) > 1:
        for i in range(1, len(sys.argv)):
            path.append(sys.argv[i])
    else:
        # path = ["p1.jpg", "p2.jpg"]
        # path = ["p1.jpg", "p2.jpg", "p3.jpg"]
        path = ["p1.jpg", "p2.jpg", "p3.jpg", "p4.jpg"]

    images = []

    for p in path:
        raw = cv2.imread(p)
        images.append(Image(resize(raw)))

    result = images[0].raw

    for i in range(1,len(images)):
        img1 = Image(result)
        img2 = images[i]
        visualizeMatch(img1, img2)
        M = transform(img1, img2)
        # print(M)
        warpImg = cv2.warpPerspective(img2.raw, np.linalg.inv(M), (img1.raw.shape[1]+img2.raw.shape[1], img1.raw.shape[0]))
        result = stitch(img1.raw, warpImg)

    cv2.imshow("Panorama", result)
    cv2.waitKey(0)
    cv2.imwrite("result.jpg", result)



