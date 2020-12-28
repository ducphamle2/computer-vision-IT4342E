
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

class Translator:

    def translateText(self, imgPath, textListDict, langCode):
        fileName=os.path.basename(imgPath)    
        textList=textListDict[fileName]
        textList_trans=[]
        for text in textList:
            # use translator lib to translate the text
            text_trans = translator.translate(text, lang_tgt=langCode,) if len(text)!=0 else "" 
            # store it into the text list to draw back again to the inpained img
            textList_trans += [text_trans]
        return textList_trans