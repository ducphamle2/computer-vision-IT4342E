# computer-vision-IT4342E
Computer Vision midterm project ICT-01.K61 

### OCR, and why this project is an OCR system: 

Because we need to extract texts from a 2D image into a text form that the machine can understand before translating into a different language. Finally, we need to transform such information back into the image text representation. The whole process has all properties of an OCR system, where it contains several subprocesses including localizing text, character segmentation and recognition. We also need to have some image pre-processing and post-processing steps as well.

### Tools and libraries needed to complete the project

**a)**: [SickZil-Machine](https://github.com/KUR-creative/SickZil-Machine) - an open source helper tool that automates text removal from conversation text boxes of manga/comics. This is a preprocessing step which uses Neural network to process the images (need to understand this) => use tensorflow.

**b)**: [Tensorflow](https://www.tensorflow.org/) - a deep learning library that will be used to train our model. Why do we need to use Tensorflow ? Maybe because we want to optimize the training process, since choosing a good descriptor and filter may be quite troublesome with inexperienced developers.

**c)**: [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) - an open source OCR Engine focusing on character patterns recognition. There are other open source libraries that target this field like [SwiftOCR](https://github.com/NMAC427/SwiftOCR), but Tesseract is the most popular one with many tutorials and documentations online. As a result, we chose Tesseract for our project. Modern Tesseract behind the scene uses LSTM - a form of Recurrent Neural Network to recognize a sequence of characters in an arbitrary length. On the other hand, Legacy Tesseract contains some steps to 

**d)**: Google translator using Python - We need this in order to translate the Japanese texts into different languages.

**e)**: [Text detection](https://github.com/qzane/text-detection) - an open source library to detect texts in an image (putting text into an rectangle)

**f)**: Other libraries to support the project such as matplotlib, numpy, textwrap, ...

### Flow to run the project:

**1)**: Import neccessary tools and libraries

**2)**: Download manga pages from an URL and store them into a directory (also create new directories for storing in painted and translated pages)

**3)**: Image segmentation to collect only texts in a manga page

**4)**: Text detection 

**5)**: OCR using Tesseract to convert collected texts into a machine-readable form (NLP part here)

**6)**: Translate the collected texts

**7)**: Draw the text