from google_trans_new import google_translator  
translator = google_translator()
import os
import cv2

class Translator:

    def translateText(self, fileName, textListDict, langCode):
        textList=textListDict[fileName]
        textList_trans=[]
        for text in textList:
            # use translator lib to translate the text
            text_trans = translator.translate(text, lang_tgt=langCode,) if len(text)!=0 else "" 
            # store it into the text list to draw back again to the inpained img
            textList_trans += [text_trans]
        return textList_trans