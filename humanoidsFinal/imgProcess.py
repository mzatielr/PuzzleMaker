import numpy as np
import math
import cv2
from matplotlib import pyplot as plt

DEBUG = False

def binarize(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    thresh, img_bw = cv2.threshold(blur,128,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return img_bw

def getContours(img_bw_destroy):
    contours, idk = cv2.findContours(img_bw_destroy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    h, w = img_bw_destroy.shape

    best = (0, contours[0], h+w)
    for i, c in enumerate(contours):
        M = cv2.moments(c)

        # get centroid of contour
        if M['m00'] != 0:
            centroid_x = int(M['m10']/M['m00'])
            centroid_y = int(M['m01']/M['m00'])
            dist = math.sqrt((centroid_x-(w/2))**2 + (centroid_y-(h/2))**2)
            if dist < best[2]:
                best = (i, c, dist)

    if DEBUG:
        cv2.drawContours(img_bw_destroy, contours[best[0]:best[0]+1], -1, (255, 0, 0), 3)
        cv2.imshow('bw', img_bw_destroy)
        cv2.waitKey(0)
        cv2.destroyWindow("bw")

    # assumes longest contour is puzzle piece
    return best[1]

def fitSide(img, lineStart, lineEnd, contour):

    def insert(longest, p1, p2, c):
        i = 0
        longest[0] = (p1, p2, c)
        while i < (len(longest)-1) and longest[i][2] < longest[i+1][2]:
            temp = longest[i+1]
            longest[i+1] = longest[i]
            longest[i] = temp
            i += 1

    def mag(v):
        return math.sqrt(v[0]**2 + v[1]**2)

    def pToLDist(v1):
        dot = v1[0]*vLine[0] + v1[1]*vLine[1]
        v1Mag = mag(v1)
        theta = math.acos(float(dot)/(v1Mag*lineLen))
        dist = v1Mag*math.sin(theta)
        return dist
    
    def cost(p1, p2):
        # fit line and get slope
        # contourSeg = contour[starti:(endi+1)]
        
        # [vx, vy, x, y] = cv2.fitLine(contourSeg, cv2.cv.CV_DIST_L2, 0, 0.01, 0.01)
        # print x1, y1, x2, y2
        bot = p2[0]-p1[0]
        if bot == 0:
            bot = 0.0001

        fitSlope = abs(float(p2[1]-p1[1])/bot)

        contourLen = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        slopePenalty = abs(fitSlope-desiredSlope)
        lenPenalty = (lineLen/contourLen)**3
        
        # find distance between bounding box line and fit line
        leftPt = p1
        rightPt = p2
        if p1[0] > p2[0]:
            leftPt = p2
            rightPt = p1

        v1 = (p1[0]-lineLeft[0], p1[1]-lineLeft[1])
        v2 = (p2[0]-lineLeft[0], p2[1]-lineLeft[1])

        d1 = pToLDist(v1)
        d2 = pToLDist(v2)
        distPenalty = max(d1, d2)

        h = 2.0*slopePenalty+50*lenPenalty+distPenalty
        return h

    # make sure slope doesn't divide by 0
    bot = -1

    if lineEnd[0] == lineStart[0]:
        bot = 0.0001
    else:
        bot = lineEnd[0]-lineStart[0]
    desiredSlope = abs(float(lineEnd[1]-lineStart[1])/bot)

    lineLeft = lineStart  
    lineRight = lineEnd
    if lineLeft[0] > lineRight[0]:
        lineLeft = lineEnd
        lineRight = lineStart
    vLine = (lineRight[0]-lineLeft[0], lineRight[1]-lineLeft[1])

    lineLen = mag(vLine) # math.sqrt((vLine[0])**2 + (vLine[1])**2)
    epsilon = 1.0
    numIncr = len(contour)
    print "points", lineStart, lineEnd
    print "desiredSlope", desiredSlope

    # startIndex, endIndex, cost
    # keep this sorted in decreasing order
    longest = [(0, 0, 500)] * 5

    for i in xrange(0, len(contour)):
        for j in xrange(1, numIncr):
            endi = (i+j) % len(contour)
            p1 = (contour[i][0][0], contour[i][0][1]) # min(i, endi)
            p2 = (contour[endi][0][0], contour[endi][0][1]) # max(i, endi)
            """
            if DEBUG:
                cv2.line(img, p1, p2, (0, 255, 255), 2)
                cv2.imshow("frame", img)
                cv2.waitKey(0)
                cv2.destroyWindow("frame")
            """
            # if cost(p1, p2) > epsilon:
            #     break
            c = cost(p1, p2)
            if longest[0][2] > c:
                insert(longest, i, endi, c)
    """
    if DEBUG:
        #for i in xrange(len(longest)):
            p1 = longest[-1][0]
            p2 = longest[-1][1]
            cv2.line(img, (contour[p1][0][0], contour[p1][0][1]), (contour[p2][0][0], contour[p2][0][1]), (0, 255, 0), 2)
            cv2.imshow("frame", img)
            cv2.waitKey(0)
            cv2.destroyWindow("frame")
    print longest
    """
    return longest

def chooseSides(tups, hullPts, contour):
    # maximize coverage of the contour
    contourSegs = []
    final = []
    # for i in xrange(1, len(tups)-1):
    #     if tups[i-1][1] != tups[i][0] and tups[i+1][0] != tups[i][1]:
    #        newTup = (tups[i-1][1], tups[i+1][0], tups[i])
    #        tups[i] = newTup

    print tups

    for i, t in enumerate(tups):
        oldstart = hullPts[t[0]][0]
        oldend = hullPts[t[1]][0]
        start = min(oldstart, oldend)
        end = max(oldstart, oldend)
        wrap = len(contour) - max(start, end) + min(start, end)
        print start, end
        if abs(start-end) < wrap:
            print "case 1"
            contourSegs.append(contour[start:end+1])
        else:
            print "case 2"
            endC = np.concatenate([contour[start:len(contour)], contour[0:end+1]])
            contourSegs.append(endC)

    return contourSegs


def getSegs(img):
    # fname = 'redPiece.jpg'
    # img = cv2.imread(fname) # pieces/2014-05-15 12.27.07.jpg')

    # binarize image
    img_bw = binarize(img)

    # invert the binarized image for better contour processing
    # img_bw = cv2.bitwise_not(img_bw)
    img_bw_destroy = img_bw.copy()

    # cv2.imshow("eh", img_bw)
    # cv2.waitKey(0)
    # cv2.destroyWindow("eh")

    puzzContour = getContours(img_bw_destroy)
    hull = cv2.convexHull(puzzContour)
    hullPts = cv2.convexHull(puzzContour, returnPoints=False)

    rect = cv2.minAreaRect(puzzContour)
    box = cv2.cv.BoxPoints(rect)

    # rounds point values
    box = np.int0(box)

    # assume box points are in counterclockwise order
    possibleSides = []
    for i, point in enumerate(box):
        lineStart = point
        lineEnd = 0 
        if i < len(box)-1:
            lineEnd = box[i+1]
        else: 
            lineEnd = box[0]
        if DEBUG:
            cv2.line(img, (lineStart[0], lineStart[1]), (lineEnd[0], lineEnd[1]), (255, 0, 0), 2)
            cv2.imshow("frame", img)
            cv2.waitKey(0)
            cv2.destroyWindow("frame")
        possibleSides.append(fitSide(img, lineStart, lineEnd, hull))

    # assume the sides go: bot, left, top, right
    split = [li[-1] for li in possibleSides] 

    contourSegs = chooseSides(split, hullPts, puzzContour)

    if DEBUG:
        for i in xrange(len(contourSegs)):
            print contourSegs[i][0]
            cv2.line(img, (contourSegs[i][0][0][0], contourSegs[i][0][0][1]), (contourSegs[i][-1][0][0], contourSegs[i][-1][0][1]), (0, 255, 0), 2)
            cv2.imshow("frame", img)
            cv2.waitKey(0)
            cv2.destroyWindow("frame")

    return contourSegs

def isLine(c):
    epsilon = 50
    dist = 0

    def mag(v):
        return math.sqrt(v[0]**2 + v[1]**2)

    def pToLDist(v1, vLine):
        dot = v1[0]*vLine[0] + v1[1]*vLine[1]
        v1Mag = mag(v1)
        print v1
        print 'asdasdasda'
        theta =1
        try:
            theta = math.acos(float(dot)/v1Mag)
        except:
            print v1

        dist = v1Mag*math.sin(theta)
        return dist

    [vx, vy, x, y] = cv2.fitLine(c, cv2.cv.CV_DIST_L2, 0, 0.01, 0.01)

    for p in c:
        px = p[0][0]
        py = p[0][1]
        v1 = (py-y, px-x)
        dist += pToLDist(v1, (vx, vy))

    print dist
    return dist < epsilon

def inOrOut(c, d):
    # number of points protruding
    count = 0
    thresh = 10

    if d == "N":
        for p in c:
            if p[0][1] < c[0][0][1]:
                count += 1
    elif d == "S":
        for p in c:
            if p[0][1] > c[0][0][1]:
                count += 1
    elif d == "E":
        for p in c:
            if p[0][0] > c[0][0][0]:
                count += 1
    elif d == "W":
        for p in c:
            if p[0][0] < c[0][0][0]:
                count += 1

    if count > thresh:
        return "Out"

    return "In"
