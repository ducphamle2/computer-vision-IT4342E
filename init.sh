mkdir ocr_manga/executables/

cd ocr_manga/executables/

sudo apt-get install -y fonts-nanum fonts-thai-tlwg fonts-tibetan-machine fonts-kacst fonts-khmeros-core fonts-guru fonts-lao fonts-lklug-sinhala fonts-sil-padauk fonts-sil-abyssinica fonts-takao-pgothic  fonts-dejavu-core fonts-freefont-ttf fonts-liberation ttf-ubuntu-font-family

rm -r -f "$PWD/tmp_images/"

gallery-dl "https://mangadex.org/chapter/826438"

git clone https://github.com/KUR-creative/SickZil-Machine.git

wget https://github.com/KUR-creative/SickZil-Machine/releases/download/v0.1.1-pre0/model.zip

unzip -o model.zip

cd ../../

# since the SickZil machine repo hard code the path, so we need to do this
cp -rf cnet ../resource

cp -rf snet ../resource

# pip install -r requirements.txt

# pip install google_trans_new

# pip install gallery-dl

