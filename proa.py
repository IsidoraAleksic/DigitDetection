import cv2
import numpy as np
from scipy import ndimage
from sklearn.neighbors import KNeighborsClassifier
from skimage.measure import label
from skimage.measure import regionprops
import vector
import math
import time
import os
from skimage import color
from planar import BoundingBox
from sklearn.datasets import fetch_mldata

lineR = [(100, 450), (500, 100)]
lineG = [(100, 450), (400, 100)]


def scale_to_range(image):
    return image / 255


def matrix_to_vector(image):
    return image.flatten()


def convert_output(outputs):
    return np.eye(len(outputs))


def nextId():
    global cc
    cc += 1
    return cc


def dot(v, w):
    x, y = v
    X, Y = w
    return x * X + y * Y


def length(v):
    x, y = v
    return math.sqrt(x * x + y * y)


def vector(b, e):
    x, y = b
    X, Y = e
    return (X - x, Y - y)


def unit(v):
    x, y = v
    mag = length(v)
    return (x / mag, y / mag)


def distance(p0, p1):
    return length(vector(p0, p1))


def scale(v, sc):
    x, y = v
    return (x * sc, y * sc)


def add(v, w):
    x, y = v
    X, Y = w
    return (x + X, y + Y)


def inRange(r, item, items):
    retVal = []
    for obj in items:
        mdist = distance(item['center'], obj['center'])
        if (mdist < r):
            retVal.append(obj)
    return retVal

def pnt2line(pnt, start, end):
    line_vec = vector(start, end)
    pnt_vec = vector(start, pnt)
    line_len = length(line_vec)
    line_unitvec = unit(line_vec)
    pnt_vec_scaled = scale(pnt_vec, 1.0 / line_len)
    t = dot(line_unitvec, pnt_vec_scaled)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    nearest = scale(line_vec, t)
    dist = distance(nearest, pnt_vec)
    nearest = add(nearest, start)
    return dist, int(nearest[0]), int(nearest[1])


def detectGreenLine(edges, img_dil):
    XminG = 30000
    YminG = 30000
    YmaxG = 0
    XmaxG = 0

    isGreenImage = img_dil[:, :, 1]
    green = (isGreenImage > 220).astype(np.float32)
    green = green.astype(np.uint8)
    lines = cv2.HoughLinesP(green, 1, np.pi / 180, threshold=180, minLineLength=30, maxLineGap=1)

    lineG[0] = (lines[0][0][0], lines[0][0][1])
    lineG[1] = (lines[0][0][2], lines[0][0][3])

    if lines is not None:
        for x1, y1, x2, y2 in lines[0]:
            if x1 < XminG:
                XminG = x1
                YminG = y1
            if x2 > XmaxG:
                YmaxG = y2
                XmaxG = x2

    return XminG, YminG, XmaxG, YmaxG, lineG[0], lineG[1]


def detectBlueLine(edges, img_dil):
    XminR = 30000
    YminR = 30000
    YmaxR = 0
    XmaxR = 0

    isBlueImage = img_dil[:, :, 0]
    blue = (isBlueImage > 220).astype(np.float32)
    blue = blue.astype(np.uint8)
    lines = cv2.HoughLinesP(blue, 2, np.pi / 180, threshold=180, minLineLength=30, maxLineGap=1)

    lineR[0] = (lines[0][0][0], lines[0][0][1])
    lineR[1] = (lines[0][0][2], lines[0][0][3])
    if lines is not None:
        for x1, y1, x2, y2 in lines[0]:
            if x1 < XminR:
                XminR = x1
                YminR = y1
            if x2 > XmaxR:
                XmaxR = x2
                YmaxR = y2

    return XminR, YminR, XmaxR, YmaxR, lineR[0], lineR[1]


def image_bin(image_gs):
    ret, image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin


def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_RGB2BGR)


def dilate(image):
    kernel = np.ones((2, 2))  # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)


def erode(image):
    kernel = np.ones((2, 2))  # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)

def fill(image):
    if(np.shape(image)!=(28,28)):
        img = np.zeros((28,28))
        x = 8 - np.shape(image)[0]
        y = 8 - np.shape(image)[1]
        img[:-x,:-y] = image
        return img
    else:
        return image


def newImg(img):
    try:
        label_img = label(img)
        regions = regionprops(label_img)
        newImg = ""
        minx = 1000
        miny = 1000
        maxx = -100
        maxy = -100
        for region in regions:
            bbox = region.bbox
            if bbox[0] < minx:
                minx = bbox[0]
            if bbox[1] < miny:
                miny = bbox[1]
            if bbox[2] > maxx:
                maxx = bbox[2]
            if bbox[3] > maxy:
                maxy = bbox[3]

        height = maxx - minx
        width = maxy - miny
        newImg = np.zeros((28, 28))

        newImg[0:height, 0:width] = newImg[0:height, 0:width] + img[minx:maxx, miny:maxy]
        return newImg

    except ValueError:
        pass
# color filter
kernel = np.ones((2, 2), np.uint8)
boundaries = [
    ([230, 230, 230], [255, 255, 255])
]

cc = -1
elements = []
t = 0
counter = 0
times = []
passed_red = []
passed_green = []

nNeighbours = KNeighborsClassifier(n_neighbors=1)
mnist = fetch_mldata('MNIST original')
data = mnist.data.astype('uint8')
target = mnist.target

#file = open('out.txt','w');
processed = np.empty_like(data)#Return a new array with the same shape and type as a given array.
cap = cv2.VideoCapture('video-9.avi')
for i in range(0, len(data)):
    mnist_img = data[i].reshape(28, 28)
    data_BW = ((color.rgb2gray(mnist_img) / 255.0) > 0.88).astype('uint8')
    new_mnist_img = newImg(data_BW)
    processed[i] = new_mnist_img.reshape(784, )

nNeighbours.fit(processed, target)

"""for vidIndex in range(0, 10):
    videoName = 'video-' + str(vidIndex) + '.avi';
    VIDEO_PATH = os.path.join(os.getcwd(), videoName)
    cap = cv2.VideoCapture(VIDEO_PATH); """

while(True):
    start_time = time.time()
    ret, frame = cap.read()
    frameInd = 0
    cv2.imwrite('img.jpg', frame)
    passed = 0
    windowName = 'Preview'

    ret, img = cap.read()
    if not ret:
        break
    #(lower, upper) = boundaries[0]


    # create NumPy arrays from the boundaries
    #lower = np.array(lower, dtype="uint8")
    #upper = np.array(upper, dtype="uint8")
    lower = np.array([230,230,230])
    upper = np.array([255,255,255])
    mask = cv2.inRange(img, lower, upper)
    img0 = 1.0 * mask
    img1 = 1.0 * mask
    img0 = cv2.dilate(img0, kernel)  # cv2.erode(img0,kernel)
    img0 = cv2.dilate(img0, kernel)
    img0 = cv2.dilate(img0, kernel)
    img0 = cv2.dilate(img0, kernel)

    cv2.imwrite('img.jpg', img)
    kernel = np.ones((2, 2), np.uint8)
    imgi = load_image('img.jpg')
    gray = cv2.cvtColor(imgi, cv2.COLOR_BGR2GRAY)
    ret, imgB = cv2.threshold(gray, 200, 255,
                              cv2.THRESH_BINARY)
    img_er = erode(imgi)
    img_dil = dilate(img_er)

    labeled, nr_objects = ndimage.label(img0)  # label features in array.labeled-labeled with a diff feature
    objects = ndimage.find_objects(labeled)  # the objects that are labeled-->numbers
    edges = cv2.Canny(img, 50, 160, apertureSize=5, L2gradient=True)
    x11, y11, x22, y22, lineR[0], lineR[1] = detectBlueLine(edges, img)
    x1, y1, x2, y2, lineG[0], lineG[1] = detectGreenLine(edges, img)
    for i in range(nr_objects):  # nr_objects is the number of elements in the array
        loc = objects[i]
        (xc, yc) = ((loc[1].stop + loc[1].start) / 2,
                    (loc[0].stop + loc[0].start) / 2)
        (dxc, dyc) = ((loc[1].stop - loc[1].start),
                      (loc[0].stop - loc[0].start))

        if (dxc > 11 or dyc > 11):
            cv2.circle(img, (xc, yc), 16, (25, 25, 255), 1)
            elem = {'center': (xc, yc), 'size': (dxc, dyc), 't': t, 'passRed': False, 'passGreen': False, 'image': img}
            # find in range
            lst = inRange(20, elem, elements)
            nn = len(lst)
            if nn == 0:
                elem['id'] = nextId()
                elem['t'] = t
                elem['pass'] = False
                elem['history'] = [{'center': (xc, yc), 'size': (dxc, dyc), 't': t}]
                elem['future'] = []
                elements.append(elem)
            elif nn == 1:
                lst[0]['center'] = elem['center']
                lst[0]['t'] = t
                lst[0]['history'].append({'center': (xc, yc), 'size': (dxc, dyc), 't': t})
                lst[0]['future'] = []
    for el in elements:
        tt = t - el['t']
        if (tt < 3):
            c = (0, 0, 255)
            resultR = pnt2line(el['center'], lineR[0], lineR[1])
            resultG = pnt2line(el['center'], lineG[0], lineG[1])
            distR = resultR[0]
            distG = resultG[0]

            pnt1 = resultR[1]
            pnt2 = resultG[1]
            r1 = 0
            r2 = 0
            c = (0, 255, 160)
            if (distR < 8):
                if el['passRed'] == False:
                    el['passRed'] = True
                    print('passed red line')
                    xcenter = el['center'][0]
                    ycenter = el['center'][1]
                    elem['image'] = img1[ycenter - 14:ycenter + 14, xcenter - 14:xcenter + 14]
                    cv2.imshow('RED',elem['image'])
                    passed_red.append(elem['image'])

            if (distG < 9):
                if el['passGreen'] == False:
                    el['passGreen'] = True
                    print('passed green line')
                    xcenter = el['center'][0]
                    ycenter = el['center'][1]
                    elem['image'] = img1[ycenter - 14:ycenter + 14, xcenter - 14:xcenter + 14]
                    cv2.imshow('GREEN', elem['image'])
                    passed_green.append(elem['image'])

            cv2.circle(img, el['center'], 16, c, 2)

            id = el['id']

            for hist in el['history']:
                ttt = t - hist['t']

            for fu in el['future']:
                ttt = fu[0] - t

    elapsed_time = time.time() - start_time
    times.append(elapsed_time * 1000)

    # print nr_objects
    t += 1

    cv2.line(img_dil, (x1, y1), (x2, y2), (230, 255, 230), 2)
    cv2.line(img_dil, (x11, y11), (x22, y22), (230, 255, 230), 2)
    cv2.imshow('frame', img_dil)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break


cap.release()
cv2.destroyAllWindows()
s = 0
print(s)
print('RED:')
for ele in passed_red:
    newImage = newImg(ele)
    image_size = fill(np.array(newImage.astype('uint8')))
    k = nNeighbours.predict(image_size.reshape(1, -1))
    s = s + k
    # print('Prvi knn: %d' %predictKNN(newImage))

print('GREEN:')
for ele in passed_green:
    newImage = newImg(ele)
    image_size = fill(np.array(newImage.astype('uint8')))
    k = nNeighbours.predict(image_size.reshape(1, -1))
    s = s - k
    # print('Prvi knn: %d' %predictKNN(newImage))
print('KNN SUM')
print(s)
print(s[0])

et = np.array(times)
print 'mean %.2f ms' % (np.mean(et))
# print np.std(et)
#file.write(videoName + '\t' + s[0] + '\n');
#file.close();