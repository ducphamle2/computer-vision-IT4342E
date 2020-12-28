
from google_trans_new import google_translator  
translator = google_translator()
import tensorflow as tf
import sys
import os
import requests

executablePath=os.path.join(os.getcwd(), "executables/")
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
import io
from apiclient.http import MediaFileUpload, MediaIoBaseDownload
import urllib.request

########################### gooogle vision ocr
from google.oauth2 import service_account
credentials = service_account.Credentials.from_service_account_file('Credentials/vision_key.json')

GOOGLE_CLOUD_PROJECT = 'comvis-manga-translator'
from google.cloud import vision
from google.cloud.vision_v1 import types
ocr_client = vision.ImageAnnotatorClient(credentials=credentials)

class MangaDrawer:

    #################get font
    def getFont(self,lang,size=25):
        fontList=os.popen('fc-list :lang='+lang+' | grep style=Regular').read().split("\n")[:-1]  #load regular style font pathList
        if len(fontList)==0: fontList=os.popen('fc-list :lang='+self.langCode).read().split("\n")[:-1]   #if no regular style font load remain font pathList
        fontList=[i.split(":")[0] for i in fontList]              #get only path data from string
        fontPath=fontList[0]
        return ImageFont.truetype(fontPath, size)

    #################draw text
    def drawText(self,imgPath,rect,textList,lang,break_long_words=False):
        img = Image.open(imgPath)
        #fontSize=int(img.size[1]*0.008)
        #imageFont=getFont(lang,fontSize)

        draw = ImageDraw.Draw(img)
        for text,(x,y,w,h)  in zip(textList,rect):
            if text=="": continue
            #dynamic fontsize scaling
            #fontsize = rect width * 0.13
            fontSize = int(w * 0.06)
            if(fontSize < 18): 
                fontSize = 18  
            imageFont=self.getFont(lang,fontSize)
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

    
    def draw(self, imgPath, transalatedFolder, inpaintedFolder, langCode, rectDict, textListDict_trans):
        fileName=os.path.basename(imgPath)
        rectP,rect=rectDict[fileName]
        im=self.drawText(inpaintedFolder+fileName,rect,textListDict_trans[fileName], langCode)

        # another translated folder
        tranFolder= "../../computer-vision-IT4342E-FE/src/components/tmp_images/"
        im.save(tranFolder+fileName)
        im.save(transalatedFolder + fileName)

        files = {'media': open(transalatedFolder + fileName, 'rb')}
        return files