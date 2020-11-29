import logging
import time
import re
import hashlib

from includes.utils import *
from const.message import *
from config import *
from errors import *
from artificial_intelligence.ocr_anime.handler import OCRAnimeHandler

LOGGER = logging.getLogger(__name__)


class RouteHandler:

    def __init__(self):
        self.ocr_anime_translator = OCRAnimeHandler()

    async def translate(self, request):
        data = await request.post()

        # collect the image from the user
        user_image = data['image'].file

        if user_image is None:
            return fail(INVALID_INPUT)

        # read face content
        uploaded_image_content = user_image.read()

        time_stamp = int(time.time())
        images_directory = f'{OCRAnimeConfig.CORE_DATA_DIR}/db'

        # store face images
        file_name = f'{time_stamp}.jpg'

        file_path = '/'.join([images_directory, file_name])

        store_content_to_file(uploaded_image_content, file_path)

        return success({"response_data": "abc"})