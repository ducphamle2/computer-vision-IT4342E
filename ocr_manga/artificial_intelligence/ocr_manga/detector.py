import os
import cv2

class TextDetector:

    ############################text cropping rectangle
    # images passed in are the text only images, with black texts and white background.
    def text_detect(self, img, ele_size=(8,2)): #
        if len(img.shape)==3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # why do we use Sobel here but not canny ?
        img_sobel = cv2.Sobel(img,cv2.CV_8U,1,0)#same as default,None,3,1,0,cv2.BORDER_DEFAULT)
        # turn everything in the image to maximum black & maximum white. < threshold = black, > thres = white
        # must be gray img, a way to make the image clearer. best for collecting black texts.
        # OTSU is a way to choose the threshold automatically as an extra flag to a canon way of choosing
        # here we choose BINARY as canon, and OTSU as extra to support choosing threshold
        img_threshold = cv2.threshold(img_sobel,0,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY)
        
        # this is to generate a box to cover a region of a text.
        # we use rectangle shape (MORPH_RECT)
        element = cv2.getStructuringElement(cv2.MORPH_RECT,ele_size)

        # needs to read more
        img_threshold = cv2.morphologyEx(img_threshold[1],cv2.MORPH_CLOSE,element)
        
        # find the contours (đường viền) of each object. Here they are texts 
        res = cv2.findContours(img_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if cv2.__version__.split(".")[0] == '3':
            _, contours, hierarchy = res
        else:
            contours, hierarchy = res
        #no padding, box    #x,y,w,h

        # collect an array of rectangles for each contour if its row is larger than 100
        Rect = [cv2.boundingRect(i) for i in contours if i.shape[0]>100]                                              
        #with padding, box  x1,y1,x2,y2
        # padding is used to distance the box away from the text a bit innerly
        # (x1,y1) as the top-left vertex and (x2,y2) as the bottom-right vertex of a rectangle region
        RectP = [(max(int(i[0]-10),0),max(int(i[1]-10),0),min(int(i[0]+i[2]+5),img.shape[1]),min(int(i[1]+i[3]+5),img.shape[0])) for i in Rect]       
        return RectP,Rect

    def detectText(self, downloadFileList, textOnlyFolder, imgPath):
        fileName=os.path.basename(imgPath)
        img = cv2.imread(textOnlyFolder+fileName)
        #0.011 = size of textbox detected relative to img size (eg 1920 * 0.011 x 1080 * 0.011)
        # why choose 0.011 ?
        rectP, rect = self.text_detect(img, ele_size=(int(img.shape[1] * 0.02), int(img.shape[0] * 0.02)))
        return rectP, rect, fileName