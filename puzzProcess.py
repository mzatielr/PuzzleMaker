import os, sys
import cv2
from imgProcess import *

class Side:

    allDirs = ["S", "W", "N", "E"]

    def __init__(self, c, facing):
        self.contour = c

        # "Out" | "In" | "Edge"
        self.sideType = ""

        # "N" | "S" | "E" | "W"
        self.facing = self.allDirs[facing]

    def rotate(self, numTimes):
        i = self.allDirs.index(self.face)
        self.facing = (i+numTimes) % len(self.allDirs)

    def classify(self):
        myType = "Out"
        if isLine(self.contour):
            myType = "Edge"
        else:
            # find whether the side is an extrusion or an indentation; must use
            # originial image rotation
            myType = inOrOut(self.contour, self.facing)
        self.sideType = myType
        return myType


class Piece:
    
    def __init__(self, img):
        self.img = img
        self.rotation = 0

        # "Corner" | "Interior" | "Edge"
        self.pieceType = ""
        self.sides = []

    def setSides(self, contourSegs):
        for i, c in enumerate(contourSegs):
            self.sides.append(Side(c, i))

    # rotate clockwise numTimes number of times
    def rotate(self, numTimes):
        for side in self.sides:
            side.rotate(numTimes)

    def classify(self):
        count = 0
        for side in self.sides:
            tp = side.classify()
            if tp == "Exterior":
                count += 1
        if count > 2:
            print "Err: more than three exterior sides found on one piece"
        elif count == 2:
            self.pieceType = "Corner"
        elif count == 1:
            self.pieceType = "Edge"
        else:
            self.pieceType = "Interior"

def assemble(allPieces):
    return allPieces

def main():

    if len(sys.argv) < 2:
        print "Usage is python puzzProcess.py <path-to-images-folder>"
        sys.exit(0)

    folderpath = sys.argv[1]
    allPieces = []

    for f in os.listdir(folderpath):
        imgpath = os.path.join(folderpath, f)

        img = cv2.imread(imgpath)
        p = Piece(img)

        # scale images
        newx,newy = img.shape[1]/4,img.shape[0]/4
        img = cv2.resize(img,(newx,newy))

        contourSegs = getSegs(img)

        p.setSides(contourSegs)
        p.classify()

        allPieces.append(p)
    print "pass 1"
    print allPieces
    assemble(allPieces)

if __name__ == '__main__':
    main()
