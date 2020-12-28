import os
import sys
import os
executablePath=os.path.join(os.getcwd(), "executables/")
sys.path.append(os.path.join(executablePath, "SickZil-Machine/src"))
import core
import imgio    #for ez img reading and writing 
import utils.fp as fp
import cv2
from tqdm import tqdm                         #progressbar when run loop    


class ImageSegmentation:


    ################################image segmentation
    def imgpath2mask(self, imgpath):
        return fp.go(
            imgpath,
            lambda path: imgio.load(path, imgio.NDARR),     
            core.segmap,
            imgio.segmap2mask)
    
    def segmentImage(self, downloadFileList, inpaintedFolder, textOnlyFolder):
        # image segmentation
        for i, imgPath in enumerate(tqdm(downloadFileList)):
            print("image path: ", imgPath)
            fileName=os.path.basename(imgPath)
            oriImage = imgio.load(imgPath, imgio.IMAGE)                      #ori image
            # imgpath2mask(imgPath)
            # mask image is a black image with features (text) being white
            maskImage  = imgio.mask2segmap(self.imgpath2mask(imgPath))            #mask image
            # remove all the text from the original image
            # this is used later on to write new translated texts on it.
            inpaintedImage = core.inpainted(oriImage, maskImage)             #notext image
            
            # convert all the texts from white to black color (white-white => white, else black)
            # we need to convert to black color text for what ?
            # for easier reading and easier for oct algo to detect text
            textOnlyImage= cv2.bitwise_and(oriImage,maskImage)               #text only image
            #if i==0:
                #cv2.imshow("text_only_img", textOnlyImage)
            textOnlyImage[maskImage==0] = 255                     
            imgio.save(inpaintedFolder+fileName, inpaintedImage)
            imgio.save(textOnlyFolder+fileName, textOnlyImage)