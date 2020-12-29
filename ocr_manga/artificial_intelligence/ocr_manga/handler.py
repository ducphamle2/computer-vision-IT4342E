import sys
import os
import requests
import glob                                    #list path
from tqdm import tqdm                         #progressbar when run loop
from apiclient.http import MediaFileUpload, MediaIoBaseDownload
from artificial_intelligence.ocr_manga.segmentation import ImageSegmentation
from artificial_intelligence.ocr_manga.detector import TextDetector
from artificial_intelligence.ocr_manga.ocr_file import TextOCR
from artificial_intelligence.ocr_manga.translator import Translator
from artificial_intelligence.ocr_manga.drawer import MangaDrawer

class OCRMangaHandler:

  selectedLang = 'vietnamese'

  translated_list = []

  LANGUAGES = {'afrikaans': 'af', 'albanian': 'sq', 'amharic': 'am', 'arabic': 'ar', 'armenian': 'hy', 'azerbaijani': 'az', 'basque': 'eu', 'belarusian': 'be', 'bengali': 'bn', 'bosnian': 'bs', 'bulgarian': 'bg', 'catalan': 'ca', 'cebuano': 'ceb', 'chichewa': 'ny', 'chinese (simplified)': 'zh-cn', 'chinese (traditional)': 'zh-tw', 'corsican': 'co', 'croatian': 'hr', 'czech': 'cs', 'danish': 'da', 'dutch': 'nl', 'english': 'en', 'esperanto': 'eo', 'estonian': 'et', 'filipino': 'tl', 'finnish': 'fi', 'french': 'fr', 'frisian': 'fy', 'galician': 'gl', 'georgian': 'ka', 'german': 'de', 'greek': 'el', 'gujarati': 'gu', 'haitian creole': 'ht', 'hausa': 'ha', 'hawaiian': 'haw', 'hebrew': 'iw', 'hindi': 'hi', 'hmong': 'hmn', 'hungarian': 'hu', 'icelandic': 'is', 'igbo': 'ig', 'indonesian': 'id', 'irish': 'ga', 'italian': 'it', 'japanese': 'ja', 'javanese': 'jw', 'kannada': 'kn', 'kazakh': 'kk', 'khmer': 'km', 'korean': 'ko', 'kurdish (kurmanji)': 'ku', 'kyrgyz': 'ky', 'lao': 'lo', 'latin': 'la', 'latvian': 'lv', 'lithuanian': 'lt', 'luxembourgish': 'lb', 'macedonian': 'mk', 'malagasy': 'mg', 'malay': 'ms', 'malayalam': 'ml', 'maltese': 'mt', 'maori': 'mi', 'marathi': 'mr', 'mongolian': 'mn', 'myanmar (burmese)': 'my', 'nepali': 'ne', 'norwegian': 'no', 'pashto': 'ps', 'persian': 'fa', 'polish': 'pl', 'portuguese': 'pt', 'punjabi': 'pa', 'romanian': 'ro', 'russian': 'ru', 'samoan': 'sm', 'scots gaelic': 'gd', 'serbian': 'sr', 'sesotho': 'st', 'shona': 'sn', 'sindhi': 'sd', 'sinhala': 'si', 'slovak': 'sk', 'slovenian': 'sl', 'somali': 'so', 'spanish': 'es', 'sundanese': 'su', 'swahili': 'sw', 'swedish': 'sv', 'tajik': 'tg', 'tamil': 'ta', 'telugu': 'te', 'thai': 'th', 'turkish': 'tr', 'ukrainian': 'uk', 'urdu': 'ur', 'uzbek': 'uz', 'vietnamese': 'vi', 'welsh': 'cy', 'xhosa': 'xh', 'yiddish': 'yi', 'yoruba': 'yo', 'zulu': 'zu'}

  langCode=LANGUAGES[selectedLang]

  #########################################working dir
  # path to the executable folder with Sickzil lib & images for editing
  executablePath=os.path.join(os.getcwd(), "executables/")
  mainTempFolder = os.path.join(executablePath, "tmp_images/")
  textOnlyFolder=os.path.join(mainTempFolder, "textOnly/")
  inpaintedFolder=os.path.join(mainTempFolder,"inpainted/")
  transalatedFolder = os.path.join(mainTempFolder, "translated/")
  
  # init downloadFileList class variable
  downloadFileList = []


  # this function is used to collect the list images to translate from an url
  def collect_original_images(self, url, lang):
    #clean up old output folder
    os.system("rm -r -f gallery-dl")
    os.system("rm -r -f executables/tmp_images/")
    files = glob.glob("../../computer-vision-IT4342E-FE/src/components/tmp_images/*")
    for f in files:
      os.remove(f)

    #create working dir
    for filePath in [self.textOnlyFolder, self.inpaintedFolder,self.transalatedFolder]:
      if not os.path.exists(filePath):
        os.makedirs(filePath)

    # reset the list
    self.translated_list = []
        
    # download img
    print("\nDownload Image")
    sys_cmd = "gallery-dl " + url
    os.system(sys_cmd)
    # update the download file list after downloading the image
    self.downloadFileList=glob.glob("gallery-dl/**/*.jpg",recursive = True) + glob.glob("gallery-dl/**/*.png",recursive = True) + glob.glob("gallery-dl/**/*.jpeg",recursive = True)
    #downloadFileList=glob.glob("gallery-dl/*/*/*/*")
    self.downloadFileList.sort()
    mangaName = os.path.basename(glob.glob(os.path.join("gallery-dl/*/*"))[0])
    print("\nManga title: " + mangaName)
    self.langCode=self.LANGUAGES[lang]
  
  # given an url, this function will download the file and translate it
  def translate(self, url, lang, ocr, srclang):

    # download required images from the url
    self.collect_original_images(url, lang)

    rectDict = dict()
    
    # image segmentation
    print("\nImage Segmentation")
    seg = ImageSegmentation()
    seg.segmentImage(self.downloadFileList, self.inpaintedFolder, self.textOnlyFolder)


    print("\nText bound detection")
    detector = TextDetector()
    # enumerate through the list of text only images
    for i,imgPath in enumerate(tqdm(self.downloadFileList)):
      rectP,rect,fileName = detector.detectText(self.downloadFileList, self.textOnlyFolder, imgPath)  #x,y  20,25
      # each file has a a 2D array, rectP - an array of rectangle padding & rect - an array of rectangles
      rectDict[fileName]=[rectP,rect]
      # #display first page
      # if i==0:
      #   for i in rectP:
      #     cv2.rectangle(img, i[:2], i[2:], (0, 0, 255))
          
    textListDict=dict({})

    print("\nOCR")
    ocrText = TextOCR()
    for i,imgPath in enumerate(tqdm(self.downloadFileList)):
      fileName=os.path.basename(imgPath)
      textList = ocrText.textToString(self.textOnlyFolder, fileName, rectDict, ocr, srclang)
      textListDict[fileName]=textList

    print("\nTranslate")
    #####################translate
    textListDict_trans = dict({})
    trans = Translator() 
    # loop through the list of text lists collected above
    for i,imgPath in enumerate(tqdm(self.downloadFileList)):
      fileName=os.path.basename(imgPath)
      textList_trans = trans.translateText(fileName, textListDict, self.langCode)
      textListDict_trans[fileName]=textList_trans

    #print text list obtain thru ocr
    #print(textListDict)

    print("\nDraw Text")
    drawer = MangaDrawer()
    for i,imgPath in enumerate(tqdm(self.downloadFileList)):
      
      files = drawer.draw(imgPath, self.transalatedFolder, self.inpaintedFolder, self.langCode, rectDict, textListDict_trans)

      res = requests.post("http://164.90.180.95:5001/api/v0/add", files=files)

      print("res: ", res.json())

      self.translated_list.append(res.json()['Hash'])

    return self.translated_list

# run python class
# handler = OCRMangaHandler()
# handler.translate('https://mangakakalot.com/chapter/xk923531/chapter_257', 'vietnamese', 'Tesseract', 'English')