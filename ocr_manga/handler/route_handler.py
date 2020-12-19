import logging
import time
import re
import hashlib
import formencode as fe

from includes.utils import *
from const.message import *
from config import *
from errors import *
from artificial_intelligence.ocr_manga.handler import OCRMangaHandler

LOGGER = logging.getLogger(__name__)


class RouteHandler:

    def __init__(self):
        self.ocr_manga_translator = OCRMangaHandler()

    async def translate(self, request):
        data = await request.post()
        fe.variabledecode.variable_decode(data, dict_char='.', list_char='-')
        print("data", data)
        print("data lang: ", data['lang'])
        print("data url: ", data['url'])

        self.ocr_manga_translator.translate(data['url'], data['lang'])
        #self.ocr_manga_translator.translate('https://mangadex.org/chapter/826437', 'vietnamese')

        # # collect the image from the user
        # user_image = data['image'].file

        # if user_image is None:
        #     return fail(INVALID_INPUT)

        # # read face content
        # uploaded_image_content = user_image.read()

        # time_stamp = int(time.time())
        # images_directory = f'{OCRMangaConfig.CORE_DATA_DIR}/db'

        # # store face images
        # file_name = f'{time_stamp}.jpg'

        # file_path = '/'.join([images_directory, file_name])

        # store_content_to_file(uploaded_image_content, file_path)

        return success({"status": 200})