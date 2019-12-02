import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

def thresholding(imgpath):
    img = cv2.imread(imgpath)
    lo = np.mean(img) * 1.15
    print(lo)
    hi = 255
    # ret, thresh1 = cv2.threshold(img, lo, hi, cv2.THRESH_BINARY)
    # ret, thresh2 = cv2.threshold(img, lo, hi, cv2.THRESH_BINARY_INV)
    # ret, thresh3 = cv2.threshold(img, lo, hi, cv2.THRESH_TRUNC)
    ret, thresh1 = cv2.threshold(img, lo, hi, cv2.THRESH_TOZERO)
    # ret, thresh5 = cv2.threshold(img, lo, hi, cv2.THRESH_TOZERO_INV)

    lo = np.mean(thresh1)*4
    print(lo)
    ret, thresh2 = cv2.threshold(thresh1, lo, hi, cv2.THRESH_TOZERO)
    titles = ['Original Image', 'TOZERO', 'TOZERO']

    images = [img, thresh1, thresh2]

    for i in range(3):
        plt.subplot(1, 3, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])

    plt.show()

    return img

def histo(imgpath):
    img = cv2.imread(imgpath)
    # create a mask
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[100:300, 100:400] = 255
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    # Calculate histogram with mask and without mask
    # Check third argument for mask
    hist_full = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_mask = cv2.calcHist([img], [0], mask, [256], [0, 256])

    plt.subplot(221), plt.imshow(img, 'gray')
    plt.subplot(222), plt.imshow(mask, 'gray')
    plt.subplot(223), plt.imshow(masked_img, 'gray')
    plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
    plt.xlim([0, 256])

    plt.show()
    return img

for imgpath in glob.glob("*.png"):
    thresholding(imgpath)
    histo(imgpath)