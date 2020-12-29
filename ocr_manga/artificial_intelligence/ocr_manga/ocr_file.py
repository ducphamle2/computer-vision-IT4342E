import os
import cv2

########################### tesseract ocr
import pytesseract
import re                 #regex 
import io

########################### gooogle vision ocr
from google.oauth2 import service_account
credentials = service_account.Credentials.from_service_account_file('Credentials/vision_key.json')

GOOGLE_CLOUD_PROJECT = 'comvis-manga-translator'
from google.cloud import vision
from google.cloud.vision_v1 import types
ocr_client = vision.ImageAnnotatorClient(credentials=credentials)

class TextOCR:

    ###########################################ocr
    def filterText(self,inputText):
        inputText = re.sub('[\\\\+/§◎*)@<>#%(&=$_\-^:;«¢~「」〃ゝゞヽヾ一●▲・ヽ÷①↓®▽■◆『£〆∴∞▼™↑←]', '', inputText)   #remove special char
        inputText = ' '.join(inputText.split())    #remove whitespace
        return inputText

    def getTextPytesseract(self,img,srclang):
        if srclang == 'jp':
            #detect jpn
            text_tesseract = pytesseract.image_to_string(img, lang="jpn+jpn_vert+Japanese+Japanese_vert")                         #ocr jpn
        else:
            #detect eng
            text_tesseract = pytesseract.image_to_string(img, lang="eng")
            text_tesseract = self.filterText(text_tesseract)
        return text_tesseract

    def getTextGoogleVisionOcr(self,img):
        tmp_file = "tmp.png"
        cv2.imwrite(tmp_file,img)
        with io.open(tmp_file, 'rb') as image_file:
            content = image_file.read()

        image = types.Image(content=content)
        response = ocr_client.text_detection(image=image)
        texts = response.text_annotations
        #cv2_imshow(img)
        string = ''
        for idx,text in enumerate(texts):
            string+=' ' + text.description
            break

        string=string.replace('\ufeff', '') 
        string=self.filterText(string)
        os.remove(tmp_file)
        #print(string)
        return string

    def textToString(self, textOnlyFolder, fileName, rectDict, ocr, srclang):
        # read text only images
        img = cv2.imread(textOnlyFolder+fileName)
        # why use this to remove noise not Gauss or mean filter ?
        img = cv2.fastNlMeansDenoisingColored(img,None,10,10)     #remove noise
        textList=[]
        rectP,rect=rectDict[fileName]
        for x1,y1,x2,y2 in rectP: 
            # Cropping the text block for giving input to OCR 
            # reference about cropping image
            # https://stackoverflow.com/questions/15589517/how-to-crop-an-image-in-opencv-using-python
            cropped = img[y1: y2, x1: x2]
            # put the cropped text box into the tesseract model to get string text
            if ocr == 'tes':
                text=self.getTextPytesseract(cropped, srclang)
                # text_nhocr=getTextNhocr(cropped,size=2)
            else:
                text=self.getTextGoogleVisionOcr(cropped)
                # add the text into the text list to ready for translation
            textList += [text]
        
        return textList