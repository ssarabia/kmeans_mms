from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
from scipy import ndimage
import matplotlib.pyplot as plt
import cv2
import numpy as np
import sys
import scipy

image_path = sys.argv[1]
image = cv2.imread(image_path)
height, width = image.shape[:2]
image = cv2.resize(image, (int(width/10), int(height/10)))
counter = {}

def convert_image_to_LAB(image):
    imageLAB = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    imageLAB_shape = imageLAB.shape
    zeros = np.zeros((imageLAB_shape[0]*imageLAB_shape[1], 2))
    index_zeros = 0
    for i in range(imageLAB_shape[0]):
        for j in range(imageLAB_shape[1]):
            zeros[index_zeros] = np.delete(imageLAB[i][j],0)
            index_zeros = index_zeros + 1
    return zeros


def paint_image(k_means, centers, image, flag):
    imageLAB = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    imageLAB_shape = imageLAB.shape
    for i in range(imageLAB_shape[0]):
        for j in range(imageLAB_shape[1]):
            features = np.delete(imageLAB[i][j],0)
            prediction = k_means.predict([features])
            true_colors = centers[prediction[0]]
            if is_gray(true_colors[0],true_colors[1]):
                imageLAB[i][j] = [0,128,128]
            else:
                imageLAB[i][j] = [170,true_colors[0],true_colors[1]]

    imageRGB = cv2.cvtColor(imageLAB, cv2.COLOR_LAB2BGR)
    cv2.imwrite('clustered.png',imageRGB)
    if flag:
        cv2.imshow("Image clustered", imageRGB)
        cv2.waitKey(0)

def is_gray(a,b):
    if a > 118 and a < 138 and b > 118 and b < 138:
        return True
    return False

def connected_comp(fname, flag):
    blur_radius = 1.0
    threshold = 50
    img = scipy.misc.imread(fname)
    imgf = ndimage.gaussian_filter(img, blur_radius)
    labeled, nr_objects = ndimage.label(imgf > threshold)
    plt.imsave('/tmp/out.png', labeled)
    if flag:
        plt.imshow(labeled)
        plt.show()
    return nr_objects

def count_color(fname, centroid,flag):
    image = cv2.imread(fname)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    shape = image.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            if image[i][j][1] == int(centroid[0]) and image[i][j][2] == int(centroid[1]):
                image[i][j] = [100,128,128]
            else:
                image[i][j] = [0,128,128]

    imageRGB = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
    imageGray = cv2.cvtColor(imageRGB, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('coloreado.png',imageGray)
    nr_objects = connected_comp('coloreado.png',flag)
    key = " ".join(str(int(x)) for x in centroid)
    counter[key] = nr_objects

def more_one_zero(counter):
    c = 0
    for key, value in counter.items():
        if value == 0:
            c += 1
    return c


image_AB = convert_image_to_LAB(image)
for k in range(2,8):
    counter = {}
    k_means = KMeans(n_clusters=k, random_state=1).fit(image_AB)
    centers = k_means.cluster_centers_
    paint_image(k_means, centers, image, False)
    for i in range(len(centers)):
        count_color('clustered.png', centers[i], False)
    if(more_one_zero(counter)>1):
        break


k= len(counter.keys())-1
print("El K optimo es de: ",k)
counter = {}
k_means = KMeans(n_clusters=k, random_state=1).fit(image_AB)
centers = k_means.cluster_centers_
paint_image(k_means, centers, image, True)
for i in range(len(centers)):
    count_color('clustered.png', centers[i], True)
for key, value in counter.items():
    print("El número de unidades de: " + str(key) + " es " + str(value))
