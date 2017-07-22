import os, sys
import cv2
import math
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
        i = self.allDirs.index(self.facing)
        facingIndex = (i+numTimes) % len(self.allDirs)
        self.facing = self.allDirs[facingIndex]

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
        self.rotation += numTimes

    def classify(self):
        count = 0
        for side in self.sides:
            # tp = side.classify()
            tp = side.sideType
            if tp == "Edge":
                count += 1
        if count > 2:
            print "Err: more than three exterior sides found on one piece"
        elif count == 2:
            self.pieceType = "Corner"
        elif count == 1:
            self.pieceType = "Edge"
        else:
            self.pieceType = "Interior"

    def getSideType(self, d):
        for s in self.sides:
            if s.facing == d:
                return s.sideType

    def hasSide(self, (d, sType)):
        for s in self.sides:
            if s.facing == d and s.sideType == sType:
                return True
        return False

def assemble(allPieces):
    print "entered assemble"
    def getBorders(soln):
        print "entered borders"
        need = []
        nxt = len(soln)
        if nxt/3 == 0:
            need.append(("N", "Edge"))
        else:
            above = soln[nxt-3]
            if above.getSideType("S") == "Out":
                need.append(("N", "In"))
            elif above.getSideType("S") == "In":
                need.append(("N", "Out"))
            else:
                need.append(("N", "In"))
                need.append(("N", "Out"))
                return need
        if nxt/3 == 2:
            need.append(("S", "Edge"))
        if nxt%3 == 0:
            need.append(("W", "Edge"))
        else:
            left = soln[-1]
            if left.getSideType("E") == "Out":
                need.append(("W", "In"))
            elif left.getSideType("E") == "In":
                need.append(("W", "Out"))
            else:
                need.append(("W", "In"))
                need.append(("W", "Out"))
                return need
        if nxt%3 == 2:
            need.append(("E", "Edge"))
        return need
   
    def canMatch(p, bord):
        # for now, assumes there isn't more than one rotation that works
        for i in xrange(4):
            match = True
            for b in bord:
                match = match and p.hasSide(b)
            if match:
                return match
            p.rotate(1)
        return False

    def assembleHelp(pieces, soln):
        print "entered help"
        if len(pieces) == 0:
            print "len is 0"
            return soln
        else:
            # get sides that need to be matched with
            bord = getBorders(soln)
            for i, p in enumerate(pieces):
                if canMatch(p, bord):
                    soln.append(p)
                    res = assembleHelp(pieces[0:i]+pieces[i+1:], soln)
                    if res != None:
                        return res 
                    # if no solution found, remove appended piece
                    soln = soln[:-1]
        return None

    # find corner to start with
    start = None
    pieces = []
    soln = []
    for i, p in enumerate(allPieces):
        print "{} {}".format("i is: ",i)
        if p.pieceType == "Corner":
            print "corcer"
            start = p
            pieces = allPieces[0:i]+allPieces[i+1:]
            break

    for i in xrange(4):
        soln = assembleHelp(pieces, [start])
        if soln != None:
            return soln
        start.rotate(1)

    print "161"
    print pieces
    return None

def showFinal(final):
    print "SHOWING FINAL !!!!"
    rows = []

    for i, piece in enumerate(final):
        img = piece.img

        angle = 90*(4-piece.rotation)
        w = img.shape[1]
        h = img.shape[0]
        scale = 0.3

        rangle = math.pi/2.0 * (4-piece.rotation)
        nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
        nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
        rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
        # calculate the move from the old center to the new center combined
        # with the rotation
        rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
        rot_mat[0,2] += rot_move[0]
        rot_mat[1,2] += rot_move[1]
        rotImg = cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)
        
        # fix slight dimension differences from scaling
        rotImg = rotImg[0:h*scale, 0:w*scale]
        if i%3 == 0:
            rows.append(rotImg)
        else:
            rows[i/3] = np.hstack((rows[i/3], rotImg))

    output = rows[0]
    for i in xrange(1, len(rows)):
        r = rows[i]
        output = np.vstack((output, r))
    
    cv2.imshow("final", output)
    cv2.waitKey(0)
    cv2.destroyWindow("final")
    return output

def main():

    if len(sys.argv) < 2:
        print "Usage is python assembly.py <path-to-images-folder>"
        sys.exit(0)

    folderpath = sys.argv[1]
    allPieces = []
    c = [None] * 4
    
    for f in os.listdir(folderpath):
        
        imgpath = os.path.join(folderpath, f)
        
        f = f[:f.index(".")]
        sTypes = f.split()

        img = cv2.imread(imgpath)

        # scale images
        newx,newy = img.shape[1]/4,img.shape[0]/4
        assert(newx == 480 and newy == 480)
        img = cv2.resize(img,(newx,newy))

        # contourSegs = getSegs(img)

        # p.setSides(contourSegs)
        # p.classify()

        p = Piece(img)
        p.setSides(c)
        for i in xrange(4):
            p.sides[i].sideType = sTypes[i]
        p.classify()

        allPieces.append(p)


    p0 = Piece(allPieces[0])
    p0.setSides(c)
    p0.sides[0].sideType = "Out"
    p0.sides[1].sideType = "Edge"
    p0.sides[2].sideType = "Edge"
    p0.sides[3].sideType = "Out"
    
    p1 = Piece(allPieces[1])
    p1.setSides(c)
    p1.sides[0].sideType = "In"
    p1.sides[1].sideType = "In"
    p1.sides[2].sideType = "Edge"
    p1.sides[3].sideType = "In"

    p2 = Piece(allPieces[2])
    p2.setSides(c)
    p2.sides[0].sideType = "Out"
    p2.sides[1].sideType = "Out"
    p2.sides[2].sideType = "Edge"
    p2.sides[3].sideType = "Edge"

    p3 = Piece(allPieces[3])
    p3.setSides(c)
    p3.sides[0].sideType = "In"
    p3.sides[1].sideType = "Edge"
    p3.sides[2].sideType = "In"
    p3.sides[3].sideType = "Out"

    p4 = Piece(allPieces[4])
    p4.setSides(c)
    p4.sides[0].sideType = "Out"
    p4.sides[1].sideType = "In"
    p4.sides[2].sideType = "Out"
    p4.sides[3].sideType = "In"
    
    p5 = Piece(allPieces[5])
    p5.setSides(c)
    p5.sides[0].sideType = "In"
    p5.sides[1].sideType = "Out"
    p5.sides[2].sideType = "In"
    p5.sides[3].sideType = "Edge"

    p6 = Piece(allPieces[6])
    p6.setSides(c)
    p6.sides[0].sideType = "Edge"
    p6.sides[1].sideType = "Edge"
    p6.sides[2].sideType = "Out"
    p6.sides[3].sideType = "Out"

    p7 = Piece(allPieces[7])
    p7.setSides(c)
    p7.sides[0].sideType = "Edge"
    p7.sides[1].sideType = "In"
    p7.sides[2].sideType = "In"
    p7.sides[3].sideType = "In"

    p8 = Piece(allPieces[8])
    p8.setSides(c)
    p8.sides[0].sideType = "Edge"
    p8.sides[1].sideType = "Out"
    p8.sides[2].sideType = "Out"
    p8.sides[3].sideType = "Edge"

    allPieces = [p0, p1, p2, p3, p4, p5, p6, p7, p8]
    # [p8, p1, p0, p3, p4, p2, p5, p6, p7]
    for p in allPieces:
        p.classify()


    print "PASS 2"

    final = assemble(allPieces)

    if final == None:
        print "No assembly possible"
    else:
        showFinal(final)

if __name__ == '__main__':
    main()
