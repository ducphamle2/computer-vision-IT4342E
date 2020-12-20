#######install cloudSDK
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
sudo apt-get install apt-transport-https ca-certificates gnupg
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
sudo apt-get update && sudo apt-get install google-cloud-sdk

pip install --upgrade google-cloud-vision
pip install google-cloud-translate==2.0.1


######run these each time start terminal to use google cloud vision ocr
export GOOGLE_APPLICATION_CREDENTIALS="Credentials/vision_key.json"