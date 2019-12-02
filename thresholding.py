import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

masks = {
         'ircam1571825132561437576.png':[200,385, 550,615],
         'ircam1571825294694103257.png':[200,565, 500,642],
         'ircam1571746715348190652.png':[195,405,600,670],
         'ircam1571746625354059903.png':[170,245,415,445],
         'ircam1571825078917436229.png':[195,400,385,460],
         '1571825186737317588_IR.png':[180,265,580,625],
         '1571825142073643588_IR.png':[200,480,0,93]
         }

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
    print(imgpath)
    img = cv2.imread(imgpath)
    # create a mask

    mask = np.zeros(img.shape[:2], np.uint8)
    y1,y2,x1,x2 = masks[imgpath][0],masks[imgpath][1],masks[imgpath][2],masks[imgpath][3]
    mask[y1:y2, x1:x2] = 255
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    # Calculate histogram with mask and without mask
    # Check third argument for mask
    hist_full = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_mask = cv2.calcHist([img], [0], mask, [256], [0, 256])

    hist_full_norm = cv2.normalize(hist_full, hist_full, 1, 0, cv2.NORM_L1)

    hist_mask_norm = cv2.normalize(hist_mask, hist_mask, 1, 0, cv2.NORM_L1)

    plt.subplot(221), plt.imshow(img, 'gray')
    plt.subplot(222), plt.imshow(mask, 'gray')
    plt.subplot(223), plt.imshow(masked_img, 'gray')
    plt.subplot(224), plt.plot(hist_full_norm), plt.plot(hist_mask_norm)
    plt.axvline(img.mean(), color='k', linestyle='dashed', linewidth=1)
    plt.xlim([0, 256])
    min_ylim, max_ylim = plt.ylim()
    plt.text(img.mean() * 1.1, max_ylim * 0.9, 'Mean: {:.2f}'.format(img.mean()))

    plt.show()
    return img

for imgpath in glob.glob("*.png"):
    # thresholding(imgpath)
    histo(imgpath)