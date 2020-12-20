import urllib.request
import glob 
import os

# path to the executable folder with Sickzil lib & images for editing
executablePath=os.path.join(os.getcwd(), "executables/")

mainTempFolder = os.path.join(executablePath, "tmp_images/")
textOnlyFolder=os.path.join(mainTempFolder, "textOnly/")
inpaintedFolder=os.path.join(mainTempFolder,"inpainted/")
transalatedFolder=os.path.join(mainTempFolder, "translated/")

#create working dir
for filePath in [textOnlyFolder,inpaintedFolder,transalatedFolder]:
  if not os.path.exists(filePath):
      os.makedirs(filePath)

#urllib.request.urlretrieve('https://www.nationalgeographic.com/content/dam/animals/thumbs/rights-exempt/reptiles/g/green-anaconda_thumb.jpg', originalTextFolder + "img.jpg")

downloadFileList=glob.glob("gallery-dl/*/*/*/*")
downloadFileList.sort()
print(downloadFileList)
#print(os.path.basename(downloadFileList[0]))