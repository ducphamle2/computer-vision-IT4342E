#https://github.com/KUR-creative/SickZil-Machine                      remove text
#https://pypi.org/project/pytesseract/                                image to text
#https://github.com/fireae/nhocr                                       image to text
#https://pypi.org/project/googletrans/                                translator


########################### translator
#https://github.com/lushan88a/google_trans_new
from google_trans_new import google_translator  
translator = google_translator()

# %tensorflow_version 1.x
import tensorflow as tf
import sys
import os

# path to the executable folder with Sickzil lib & images for editing
executablePath=os.path.join(os.getcwd(), "ocr_manga/executables/")

#print("sickZil machine path: ", os.path.join(executablePath, "SickZil-Machine/src"))
sys.path.append(os.path.join(executablePath, "SickZil-Machine/src"))
import core
import imgio    #for ez img reading and writing 
import utils.fp as fp
import cv2

########################### tesseract ocr
import pytesseract
import glob                                    #list path
from PIL import Image, ImageFont, ImageDraw   #draw text
import textwrap                               #draw text
from tqdm import tqdm                         #progressbar when run loop
from matplotlib import pyplot as plt
import re                 #regex       
from pdb import set_trace   #debug
import numpy as np

selectedLang = 'vietnamese'

LANGUAGES = {'afrikaans': 'af', 'albanian': 'sq', 'amharic': 'am', 'arabic': 'ar', 'armenian': 'hy', 'azerbaijani': 'az', 'basque': 'eu', 'belarusian': 'be', 'bengali': 'bn', 'bosnian': 'bs', 'bulgarian': 'bg', 'catalan': 'ca', 'cebuano': 'ceb', 'chichewa': 'ny', 'chinese (simplified)': 'zh-cn', 'chinese (traditional)': 'zh-tw', 'corsican': 'co', 'croatian': 'hr', 'czech': 'cs', 'danish': 'da', 'dutch': 'nl', 'english': 'en', 'esperanto': 'eo', 'estonian': 'et', 'filipino': 'tl', 'finnish': 'fi', 'french': 'fr', 'frisian': 'fy', 'galician': 'gl', 'georgian': 'ka', 'german': 'de', 'greek': 'el', 'gujarati': 'gu', 'haitian creole': 'ht', 'hausa': 'ha', 'hawaiian': 'haw', 'hebrew': 'iw', 'hindi': 'hi', 'hmong': 'hmn', 'hungarian': 'hu', 'icelandic': 'is', 'igbo': 'ig', 'indonesian': 'id', 'irish': 'ga', 'italian': 'it', 'japanese': 'ja', 'javanese': 'jw', 'kannada': 'kn', 'kazakh': 'kk', 'khmer': 'km', 'korean': 'ko', 'kurdish (kurmanji)': 'ku', 'kyrgyz': 'ky', 'lao': 'lo', 'latin': 'la', 'latvian': 'lv', 'lithuanian': 'lt', 'luxembourgish': 'lb', 'macedonian': 'mk', 'malagasy': 'mg', 'malay': 'ms', 'malayalam': 'ml', 'maltese': 'mt', 'maori': 'mi', 'marathi': 'mr', 'mongolian': 'mn', 'myanmar (burmese)': 'my', 'nepali': 'ne', 'norwegian': 'no', 'pashto': 'ps', 'persian': 'fa', 'polish': 'pl', 'portuguese': 'pt', 'punjabi': 'pa', 'romanian': 'ro', 'russian': 'ru', 'samoan': 'sm', 'scots gaelic': 'gd', 'serbian': 'sr', 'sesotho': 'st', 'shona': 'sn', 'sindhi': 'sd', 'sinhala': 'si', 'slovak': 'sk', 'slovenian': 'sl', 'somali': 'so', 'spanish': 'es', 'sundanese': 'su', 'swahili': 'sw', 'swedish': 'sv', 'tajik': 'tg', 'tamil': 'ta', 'telugu': 'te', 'thai': 'th', 'turkish': 'tr', 'ukrainian': 'uk', 'urdu': 'ur', 'uzbek': 'uz', 'vietnamese': 'vi', 'welsh': 'cy', 'xhosa': 'xh', 'yiddish': 'yi', 'yoruba': 'yo', 'zulu': 'zu'}

langCode=LANGUAGES[selectedLang]


#@markdown ---
#@markdown ### Enter url:
#url = input("Enter manga URL (https://mangadex.org/chapter/826437): ") 
url = "https://mangadex.org/chapter/826437" #@param {type:"string"}
 #@param {type:"string"}
#@markdown ---


################googleocr
import io
from apiclient.http import MediaFileUpload, MediaIoBaseDownload

###############visionOCR
GOOGLE_CLOUD_PROJECT = 'comvis-manga-translator'
from google.cloud import vision
from google.cloud.vision_v1 import types
import six
from google.cloud import translate_v2 as translate

translate_client = translate.Client()
ocr_client = vision.ImageAnnotatorClient()





#@title download from url


#########################################working dir
mainTempFolder=os.path.join(executablePath, "tmp_images/")
textOnlyFolder=os.path.join(mainTempFolder, "textOnly/")
inpaintedFolder=os.path.join(mainTempFolder,"inpainted/")
transalatedFolder=os.path.join(mainTempFolder, "translated/")

#delete if exist
os.system("rm -r -f gallery-dl")
os.system("rm -r -f executable/tmp_images/")

#create working dir
for filePath in [textOnlyFolder,inpaintedFolder,transalatedFolder]:
  if not os.path.exists(filePath):
      os.makedirs(filePath)


#############################################download jpg from site

# print("\nDownload image")

#download img
sys_cmd = "gallery-dl " + url
os.system(sys_cmd)


downloadFileList=glob.glob(os.path.join(executablePath, "gallery-dl/*/*/*/*"))
downloadFileList.sort()
mangaName = os.path.basename(glob.glob(os.path.join(executablePath, "gallery-dl/*/*"))[0])
print("\nManga title: " + mangaName)
#print(downloadFileList)
#print(os.path.basename(downloadFileList[0]))
#Image.open(downloadFileList[0])

#@title image segmentation

################################image segmentation
def imgpath2mask(imgpath):
    return fp.go(
        imgpath,
        lambda path: imgio.load(path, imgio.NDARR),     
        core.segmap,
        imgio.segmap2mask)

print("\nImage Segmentation")
for i,imgPath in enumerate(tqdm(downloadFileList)):
    fileName=os.path.basename(imgPath)
    oriImage = imgio.load(imgPath, imgio.IMAGE)                      #ori image
    # imgpath2mask(imgPath)
    # mask image is a black image with features (text) being white
    maskImage  = imgio.mask2segmap(imgpath2mask(imgPath))            #mask image
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

    #display
    #print tấm thứ 1

    #if i==0:
        #cv2.imshow("original_img", oriImage)
        #cv2.imshow("mask_img", maskImage)
        #cv2.imshow("inpainted_img", inpaintedImage)
        #cv2.imshow("text_only_img", textOnlyImage)

#@title detect text rectangle bound

############################text cropping rectangle
#https://github.com/qzane/text-detection

# images passed in are the text only images, with black texts and white background.
print("\nText bound detection")
def text_detect(img,ele_size=(8,2)): #
    if len(img.shape)==3:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
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

rectDict=dict()

# enumerate through the list of text only images
for i,imgPath in enumerate(tqdm(downloadFileList)):
    fileName=os.path.basename(imgPath)
    img = cv2.imread(textOnlyFolder+fileName)
    #0.011 = size of textbox detected relative to img size (eg 1920 * 0.011 x 1080 * 0.011)
    # why choose 0.011 ?
    rectP,rect = text_detect(img,ele_size=(int(img.shape[1]*0.02),int(img.shape[0]*0.02)))  #x,y  20,25
    # each file has a a 2D array, rectP - an array of rectangle padding & rect - an array of rectangles
    rectDict[fileName]=[rectP,rect]
    #display first page
    if i==0:
      for i in rectP:
        cv2.rectangle(img,i[:2],i[2:],(0,0,255))
      #cv2.imshow("texts with rectangles", img)

#@title OCR and translate

###########################################ocr
print("\nOCR")
def filterText(inputText):
    inputText = re.sub('[\\\\+/§◎*)@<>#%(&=$_\-^:;«¢~「」〃ゝゞヽヾ一●▲・ヽ÷①↓®▽■◆『£〆∴∞▼™↑←]', '', inputText)   #remove special char
    inputText = ' '.join(inputText.split())    #remove whitespace
    return inputText

def getTextPytesseract(img):
    #detect jpn
    #text_tesseract = pytesseract.image_to_string(img, lang="jpn+jpn_vert+Japanese+Japanese_vert")                         #ocr jpn
   
    #detect eng
    text_tesseract = pytesseract.image_to_string(img, lang="eng")
    text_tesseract = filterText(text_tesseract)
    return text_tesseract

def getTextGoogleOcr(img):
    exceptionCount=0
    while exceptionCount<5:
        try:
            #https://tanaikech.github.io/2017/05/02/ocr-using-google-drive-api/
            txtPath = 'googleocr.txt'  # Text file outputted by OCR
            imgPath="googleocr.jpg"
            cv2.imwrite(imgPath, img)  
            mime = 'application/vnd.google-apps.document'
            res = service.files().create(
                body={'name': imgPath,
                    'mimeType': mime },
                media_body=MediaFileUpload(imgPath, mimetype=mime, resumable=True) ).execute()
            downloader = MediaIoBaseDownload(
                io.FileIO(txtPath, 'wb'),
                service.files().export_media(fileId=res['id'], mimeType="text/plain"))
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            service.files().delete(fileId=res['id']).execute()
            with  open(txtPath, "r") as f:   text_google = f.read()    #txt to str
            text_google=text_google.replace('\ufeff', '') 
            text_google=filterText(text_google)
        except:
            exceptionCount+=1
            continue
        break

    return text_google

def getTextGoogleVisionOcr(img):
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
    string=filterText(string)
    os.remove(tmp_file)
    #print(string)
    return string

textListDict=dict({})
for i,imgPath in enumerate(tqdm(downloadFileList)):
    fileName=os.path.basename(imgPath)
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
      text=getTextPytesseract(cropped)
      #text=getTextGoogleVisionOcr(cropped)
      # text_nhocr=getTextNhocr(cropped,size=2)
      #text=detect_text_gg_vision(cropped)
      # add the text into the text list to ready for translation
      textList+=[text]
    textListDict[fileName]=textList

#####################translate
textListDict_trans=dict({})


print("\nTranslate")
#loop through the list of text lists collected above
for i,imgPath in enumerate(tqdm(downloadFileList)):
    fileName=os.path.basename(imgPath)    
    textList=textListDict[fileName]
    textList_trans=[]
    for text in textList:
        # use translator lib to translate the text
        text_trans=translator.translate(text, lang_tgt=langCode,)    if len(text)!=0 else "" 
        # store it into the text list to draw back again to the inpained img
        textList_trans+=[text_trans]
    textListDict_trans[fileName]=textList_trans


#print text list obtain thru ocr

#@title draw text

#################get font
def getFont(lang,size=25):
  fontList=os.popen('fc-list :lang='+lang+' | grep style=Regular').read().split("\n")[:-1]  #load regular style font pathList
  if len(fontList)==0: fontList=os.popen('fc-list :lang='+langCode).read().split("\n")[:-1]   #if no regular style font load remain font pathList
  fontList=[i.split(":")[0] for i in fontList]              #get only path data from string
  fontPath=fontList[0]
  return ImageFont.truetype(fontPath, size)


print("\nDraw Text")
#################draw text
def drawText(imgPath,rect,textList,lang,break_long_words=False):
  img = Image.open(imgPath)
  draw = ImageDraw.Draw(img)
  for text,(x,y,w,h)  in zip(textList,rect):
    if text=="": continue
    #dynamic fontsize scaling
    #fontsize = rect width * 0.13
    fontSize = int(w * 0.06)
    if(fontSize < 18): 
      fontSize = 18  
    imageFont=getFont(lang,fontSize)
    for line in textwrap.wrap(text, width=w//imageFont.size+4,break_long_words=break_long_words):   #split text to fit into box
        #text stroke
        shadowcolor=(255,255,255) #white
        strokeSize=2
        # thin border
        draw.text((x-strokeSize, y), line, font=imageFont, fill=shadowcolor)
        draw.text((x+strokeSize, y), line, font=imageFont, fill=shadowcolor)
        draw.text((x, y-strokeSize), line, font=imageFont, fill=shadowcolor)
        draw.text((x, y+strokeSize), line, font=imageFont, fill=shadowcolor)
        # thicker border
        draw.text((x-strokeSize, y-strokeSize), line, font=imageFont, fill=shadowcolor)
        draw.text((x+strokeSize, y-strokeSize), line, font=imageFont, fill=shadowcolor)
        draw.text((x-strokeSize, y+strokeSize), line, font=imageFont, fill=shadowcolor)
        draw.text((x+strokeSize, y+strokeSize), line, font=imageFont, fill=shadowcolor)
        #draw text
        draw.text((x, y), line, font=imageFont, fill=(0, 0, 0))  #black
        y += imageFont.size+strokeSize
  return img

for i,imgPath in enumerate(tqdm(downloadFileList)):
    fileName=os.path.basename(imgPath)
    rectP,rect=rectDict[fileName]
    im=drawText(inpaintedFolder+fileName,rect,textListDict_trans[fileName],langCode)
    im.save(transalatedFolder+fileName) 
    #display
    #if i==0:
      #im_oriText=drawText(inpaintedFolder+fileName,rect,textListDict[fileName],"en",break_long_words=True)
      #cv2.imshow("original img", im_oriText)
      #cv2.imshow("translated img", im)