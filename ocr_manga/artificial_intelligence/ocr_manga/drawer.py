import os
import cv2
from PIL import Image, ImageFont, ImageDraw   #draw text
import textwrap                               #draw text

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
            if(fontSize < 15): 
                fontSize = 15 
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
        #im.save(tranFolder+fileName)
        im.save(transalatedFolder + fileName)

        files = {'media': open(transalatedFolder + fileName, 'rb')}
        return files