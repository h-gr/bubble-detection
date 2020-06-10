# Copyright Georges Quénot - CNRS-LIG
# This software comes without any Guarantee
# Version 3 - 2020-03-13

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math
from scipy import interpolate as inpo
from scipy import signal
import scipy.ndimage.filters
import time
from skimage import measure

def qbubbleLabels(mask):
    return(measure.label(mask, background=0))

def qbubbleLabelAreas(labels):
    nl = labels.max()
    return(list(np.histogram(labels, bins = nl, range = (0.5, nl+0.5))[0]))

def qbubbleLabelsContours(labels):
    # Makes a list of contours from an image of labeled (numbered) regions
    lv = []
    for i in range(np.max(labels)):
        inside = np.equal(labels,i+1)
        lv.append(qbubbleInsideContour(inside))
    return(lv)

def qbubbleSplit(v):
    # Order in v is: xc, yc, a0, a1, a2, ..., aI, b1, b2, ..., bI
    I = (v.size-3)//2
    (xc, yc, a0), a, b = v[0:3], v[3:I+3], v[I+3:]
    return((xc, yc, a0, a, b, I))

def qbubblePointInside(v, x, y, m = 0):
    # Tells whether a point is inside a contour spline with a margin of m pixels
    xc, yc, r, a, b, I = qbubbleSplit(v)
    t = math.atan2(y-yc, x-xc)
    for i in range(0, I): r += a[i]*math.cos((i+1)*t)+b[i]*math.sin((i+1)*t)
    return((x-xc)**2+(y-yc)**2 < (r+m)**2)

def qbubbleImageInside(v, w, h, m = 0):
    # Builds a boolean image of points inside a contour with a margin of m pixels
    xc, yc, r, a, b, I = qbubbleSplit(v)
    dx = np.repeat(np.expand_dims((np.array(range(w))+0.5), axis = 0), h, axis = 0)-xc
    dy = np.repeat(np.expand_dims((np.array(range(h))+0.5), axis = 1), w, axis = 1)-yc
    r = np.repeat(np.expand_dims(r, axis = 0), h, axis = 0)
    r = np.repeat(np.expand_dims(r, axis = 1), w, axis = 1)
    t = np.arctan2(dy, dx)
    for i in range(0, I): r += a[i]*np.cos((i+1)*t)+b[i]*np.sin((i+1)*t)
    return(np.less(np.square(dx)+np.square(dy), np.square(r+m)))

def qbubbleImageContour(inside):
    # Create a fuzzy contour image for contour spline search from a binary image
    h, w = inside.shape
    x = np.repeat(np.expand_dims((np.array(range(w))+0.5), axis = 0), h, axis = 0)
    y = np.repeat(np.expand_dims((np.array(range(h))+0.5), axis = 1), w, axis = 1)
    a = np.sum(inside)
    xg = np.sum(x*inside)/a
    yg = np.sum(y*inside)/a
    simg = 1.0-scipy.ndimage.filters.gaussian_filter(1.0*inside, sigma=2.0)
    sdx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype = 'float32')/4
    sdy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype = 'float32')/4
    sdx_img = signal.convolve2d(simg, sdx, boundary='symm', mode='same')
    sdy_img = signal.convolve2d(simg, sdy, boundary='symm', mode='same')
    sdm_img = np.sqrt(np.square(sdx_img)+np.square(sdy_img))
    return(((1-sdm_img/np.amax(sdm_img)), xg, yg, math.sqrt(a/math.pi)))

def qbubbleIoU(v1, v2, w, h):
    # Compute IoU between delimited regions
    inside1 = qbubbleImageInside(v1, w, h)
    inside2 = qbubbleImageInside(v2, w, h)
    u = np.sum(np.logical_or(inside1, inside2))
    if (u > 0): iou = np.sum(np.logical_and(inside1, inside2))/u 
    else: iou = 0
    return(iou)


def qbubbleIoUsSquare(lv1, coord, w, h):
    iou = []
    for i in range(len(coord)):
        inside1 = qbubbleImageInside(lv1[i], w, h)
        inside2 = np.zeros((h, w), dtype = 'uint8')
        for y in range (int(coord[i][1]),int(coord[i][3])):
            for x in range (int(coord[i][0]),int(coord[i][2])):
                inside2[x][y]=1
        u = np.sum(np.logical_or(inside1, inside2))
        if (u > 0): iou.append(np.sum(np.logical_and(inside1, inside2))/u)
        else: iou.append(0)
    return(iou)
    
def qbubbleIoUs(lv1, lv2, w, h):
    # Compute IoUs between delimited regions
    iou = []
    for i in range(len(lv1)):
        inside1 = qbubbleImageInside(lv1[i], w, h)
        inside2 = qbubbleImageInside(lv2[i], w, h)
        u = np.sum(np.logical_or(inside1, inside2))
        if (u > 0): iou.append(np.sum(np.logical_and(inside1, inside2))/u)
        else: iou.append(0)
    return(iou)

def qbubbleIoULabels(labels, lv, w, h):
    # Compute IoUs between delimited regions
    iou = []
    for i in range(len(lv)):
        inside1 = np.equal(labels,i+1)
        inside2 = qbubbleImageInside(lv[i], w, h)
        u = np.sum(np.logical_or(inside1, inside2))
        if (u > 0): iou.append(np.sum(np.logical_and(inside1, inside2))/u)
        else: iou.append(0)
    return(iou)

def qbubbleArea(v, w, h):
    # Returns the area inside a contour and inside the image
    return(np.sum(qbubbleImageInside(v, w, h)))

def qbubbleAreas(lv, w, h):
    # Returns the areas inside contours and inside the image
    la = []
    for v in lv: la.append(np.sum(qbubbleImageInside(v, w, h)))
    return(la)

def qbubbleAreaF(v, m = 1):
    # Returns the area inside a contour
    xc, yc, a0, a, b, I = qbubbleSplit(v)
    s, n = 0, math.ceil(2*m*a0*math.pi)
    for k in range(n):
        tt, r = 2*k*math.pi/n, a0
        for i in range(I): r += a[i]*math.cos((i+1)*tt)+b[i]*math.sin((i+1)*tt)
        s += r**2
    return(s*math.pi/n)

def qbubbleTkPolygon(v, m = 1):
    # Returns a list of point coordinates for drawing the contour
    xc, yc, a0, a, b, I = qbubbleSplit(v)
    p, n = [], math.ceil(2*m*a0*math.pi)
    for k in range(n):
        tt, r = 2*k*math.pi/n, a0
        for i in range(I): r += a[i]*math.cos((i+1)*tt)+b[i]*math.sin((i+1)*tt)
        p.append([yc+r*math.sin(tt),xc+r*math.cos(tt)])
    return(p)

def qbubbleDraw(v, w, h, t = 0):
    # Returns a list of point coordinates for drawing the contour
    xc, yc, a0, a, b, I = qbubbleSplit(v)
    n = math.ceil(4*a0*math.pi)
    c = np.zeros((h, w), dtype = 'float32')
    for k in range(n):
        tt, r = 2*k*math.pi/n, a0
        ck, sk = math.cos(tt), math.sin(tt)
        # Contour point
        for i in range(I): r += a[i]*math.cos((i+1)*tt)+b[i]*math.sin((i+1)*tt)
        xr, yr = xc+r*ck, yc+r*sk
        # Thickness
        for i in range(-t, t+1):
            x, y = xr+0.5*i*ck, yr+0.5*i*sk
            x, y = int(math.floor(x)), int(math.floor(y))
            if (x >= 0) and (x < w) and (y >= 0) and (y < h): c[y, x] = 1
    return(np.transpose(np.array(np.where(c == 1))))

def qbubblePlot(lv, img, s = 1, t = 0, f = None, raw = np.zeros(0), rv = 1, filename = 'out'):
    # lv: list of contours' parameters
    # s: zoom factor, must be a positive integer
    # t: increased thichness of the dispayed lines
    h, w = img.shape
    tmp = np.zeros((s*h, s*w, 3), dtype = 'float32')
    # Initialize the scaled background image, pixel block version
    if f == None:
        for j in range(h):
            for i in range(w):
                for k in range(s):
                    for l in range(s):
                        for m in range(3):
                            tmp[s*j+k, s*i+l, m] = img[j, i]
    # Initialize the scaled background image, continuous version
    else:
        for i in range(s*w):
            for j in range(s*h):
                fg = f((i+0.5)/s, (j+0.5)/s)
                if fg < 0: fg = 0
                if fg > 1: fg = 1
                for m in range(3): tmp[i, j, m] = fg
    # Plot all contours
    for v in lv:
        for (y, x) in qbubbleDraw(s*v, s*w, s*h, t = t): tmp[y, x] = [1, 0, 0]
    # Plot raw contours
    if raw.size != 0:
        for j in range(h):
            for i in range(w):
                for k in range(s):
                    for l in range(s):
                        if raw[j, i] == rv: tmp[s*j+k, s*i+l] = [1, 0, 0]
    # Save the output image
    mpimg.imsave(filename+'.jpg',tmp)
    # Display the output image
    plt.imshow(tmp)
    plt.pause(0.01)

def qbubbleMaxScale(v, dv, maxmaxd):
    # Computes a scaling factor on the update so that the maximum
    # contour point displacment is bounded by maxmaxd
    dxc, dyc, da0, da, db, I = qbubbleSplit(dv)
    n, maxd = math.ceil(2*v[2]*math.pi), 0
    for k in range(n):
        tt, dr = 2*k*math.pi/n, da0
        for i in range(I):
            dr += da[i]*math.cos((i+1)*tt)+db[i]*math.sin((i+1)*tt)
        d = math.sqrt((dxc+dr*math.cos(tt))**2+(dyc+dr*math.sin(tt))**2)
        if maxd < d: maxd = d
    if maxmaxd < maxd: s = maxmaxd/maxd
    else: s = 1
    return(s)

def qbubbleGradient(v, f, w, h, p, lr, md, wd, ps):
    # Computes the gradient of the loss function
    xc, yc, a0, a, b, I = qbubbleSplit(v)
    n, m = math.ceil(2*a0*math.pi/ps), 0
    x, y = np.zeros(n, dtype = 'float32'), np.zeros(n, dtype = 'float32')
    for k in range(n):
        tt, r = 2*k*math.pi/n, a0
        for i in range(I):
            r += a[i]*math.cos((i+1)*tt)+b[i]*math.sin((i+1)*tt)
        x[k], y[k] = xc+r*math.cos(tt), yc+r*math.sin(tt)
    fx, fy = f.ev(y, x, 0, 1), f.ev(y, x, 1, 0)
    dv = np.zeros(v.size, dtype = 'float32')
    for k in range(n):
        if (x[k] > 0) and (x[k] < w) and (y[k] > 0) and (y[k] < h):
            tt = 2*k*math.pi/n
            dv[0] += fx[k]
            dv[1] += fy[k]
            fr = math.cos(tt)*fx[k]+math.sin(tt)*fy[k]
            dv[2] += fr
            for i in range(I):
                dv[i+3] += fr*math.cos((i+1)*tt) # cik terms
                dv[i+I+3] += fr*math.sin((i+1)*tt) # sik terms
            m += 1
    if (m > 0):
        dv /= m
        dv[2] -= p
        for i in range(I):
           dv[i+3] += wd*a[i]
           dv[i+I+3] += wd*b[i]
    dv *= lr
    dv *= qbubbleMaxScale(v, dv, md)
    return (dv)

def qbubbleExtend(v):
    # Add one zero-initialized harmonic component pair to a contour
    I = (v.size-3)//2
    nv = np.zeros(v.size+2, dtype = 'float32')
    nv[0:I+3], nv[I+4:2*I+4] = v[0:I+3], v[I+3:2*I+3]
    return(nv)

def qbubbleStep(v, pv, f, w, h, p, lr, th, md, wd, mi, hs, ps, rmin, vb):
    # Sequence of updates with fixed hyper-parameters
    dd = th+1
    last = -1
    for i in range(mi):
        if dd > th:
            dv = qbubbleGradient(v, f, w, h, p, lr, md, wd, ps)
            # Enforce minimum radius of rmin
            if v[2]-dv[2] < rmin: dv[2] = 0
            # Manage stopping criterion
            dd = math.sqrt(np.sum(np.square(v-0.5*dv-pv)))
            pv = v-0.5*dv
            v -= dv
            # Add more harmonics in the representations as radius increases
            if pv[2] > hs*(v.size-1)/2:
                v, pv = qbubbleExtend(v), qbubbleExtend(pv)
                if (vb > 2): print("%3d " % (i+1), pv)
            last = i
    if (vb > 1): print("%3d " % (last+1), pv)
    return((v, pv))

def qbubbleFit(x, y, f, w, h, p, lr = [100, 100, 10, 1], th = 0.001, md = 1.0,
wd = 0.0001, mi = [200,50,20,0], hs = 4, ps = 1, r = 1, rmin = 1, vb = 1):
    # x, y: coordinate of the starting point
    # f: continuous (and differentiamble) version of the original image
    # w, h: width and height of the original image
    # p: "pressure" parameter
    # lr: learning rate for gradient descent
    # th: stopping threshold
    # md: maximum contour displacement during a single iteration
    # wd: weight decay for regularization on harmonic components
    # mi: maximim number or itreations
    # hs: two more harmonic parameters for an increase of the radius by hs
    # ps: target pixel spacing in discretized shape
    # r: intial radius
    if r < rmin: r = rmin
    # v: contour parameters across iterations
    v = np.array((x, y, r), dtype = 'float32')
    # pv: last value with half updates (because of ocsillations)
    pv = v+10000 # Initial value far from v
    # first sequence just fitting a qbubble
    v, pv = qbubbleStep(v, pv, f, w, h, p, lr[0], th, md, wd, mi[0], hs, ps, rmin, vb)
    # further refinements without pressure and with decreasing lr
    for i in range(1, 4):
        v, pv = qbubbleStep(v, pv, f, w, h, 0, lr[i], th, md, wd, mi[i], hs, ps, rmin, vb)
    return(pv)

def qbubbleInsideContour(inside):
    # Extracts a contour spline from a binary image of inside pixels
    h, w = inside.shape
    contour, x, y, r = qbubbleImageContour(inside)
    f = inpo.RectBivariateSpline(np.arange(h)+0.5,np.arange(w)+0.5,contour)
    return(qbubbleFit(x, y, f, w, h, 0.04))

def qbubbleAuto(img, b = 100, p = 0.02, mi = [500,50,20,0], hs = 5, rmin = 2):
    # Automatic search of bubble contours, very slow, false positive and negatives
    # Requires manual filtering and adding of missed bubbles
    h, w = img.shape
    f = inpo.RectBivariateSpline(np.arange(h)+0.5,np.arange(w)+0.5,img)
    simg = 0.9*scipy.ndimage.filters.gaussian_filter(img, sigma=5.0)
    simg += 0.09*scipy.ndimage.filters.gaussian_filter(img, sigma=20.0)
    simg += 0.01*scipy.ndimage.filters.gaussian_filter(img, sigma=80.0)
    t = time.time()
    result = np.where(simg == np.amax(simg))
    x, y = result[1][0], result[0][0]
    inside = qbubbleImageInside(np.array((x, y ,2), dtype ='float32'), w, h, m = 1)
    simg -= scipy.ndimage.filters.gaussian_filter(1.0*inside, sigma=2.0)
    simg = np.maximum(simg,0)
    print(0, x, y)
    v = qbubbleFit(x, y, f, w, h, p, mi = mi, hs = hs, rmin = rmin)
    lv = [v]
    for i in range(1, b):
        inside = qbubbleImageInside(v, w, h, m = 1)
        simg -= scipy.ndimage.filters.gaussian_filter(1.0*inside, sigma=2.0)
        simg = np.maximum(simg,0)
        result = np.where(simg == np.amax(simg))
        x, y = result[1][0], result[0][0]
        inside = qbubbleImageInside(np.array((x, y ,2), dtype ='float32'), w, h, m = 1)
        simg -= scipy.ndimage.filters.gaussian_filter(1.0*inside, sigma=2.0)
        simg = np.maximum(simg,0)
        print((i+1), x, y)
        v = qbubbleFit(x, y, f, w, h, p, mi = mi, hs = hs, rmin = rmin)
        lv.append(v)
    print("Elapsed time: ",time.time()-t)
    return(lv)

def qbubbleRefineContours(lv1, img, vb = 0):
    h, w = img.shape
    # Refines contours according to an image
    f = inpo.RectBivariateSpline(np.arange(h)+0.5,np.arange(w)+0.5,img)
    lv2 = []
    if (vb > 0): t0 = time.time()
    for i in range(len(lv1)):
        if (vb > 1): t1 = time.time()
        v = lv1[i].copy()
        if (vb > 1): print(i, end =" ")
        pv = v+10000
        v, pv = qbubbleStep(v, pv, f, w, h, 0, 1, 0.001, 1, 0.0001, 200, 4, 1, 1, 1)
        lv2.append(v)
        if (vb > 1): print("%.2f" % (time.time()-t1))
    if (vb > 0): print("Elapsed: ",time.time()-t0)
    return(lv2)