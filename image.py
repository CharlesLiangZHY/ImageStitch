import cv2
import numpy as np

class Image:
    def __init__(self, raw):
        detector = cv2.xfeatures2d.SIFT_create()
        self.raw = raw
        self.kp, self.des = detector.detectAndCompute(cv2.cvtColor(self.raw, cv2.COLOR_BGR2GRAY),None)


def resize(raw):
    h,w,_ = raw.shape
    r = w // 1000 + 1 if w > 1000 else 1
    return cv2.resize(raw,(int(raw.shape[1]/r), int(raw.shape[0]/r)))


def visualizeMatch(img1, img2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(img1.des, img2.des, k=2)
    matched = []
    for m,n in matches:
        if m.distance < 0.5 * n.distance: # the smaller the ratio is , the correct the matches are
            matched.append([m])
    # reference: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
    demo = cv2.drawMatchesKnn(img1.raw, img1.kp, img2.raw, img2.kp, matched, None, flags=2)
    cv2.imshow("BFmatch", demo)
    cv2.waitKey(0)
    # cv2.imwrite("match.jpg", demo)
