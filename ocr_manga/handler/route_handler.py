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
        print("data srclang: ", data['srclang'])
        print("data lang: ", data['lang'])
        print("data url: ", data['url'])
        print("data ocr: ", data['ocr'])

        translated_list = self.ocr_manga_translator.translate(data['url'], data['lang'], data['ocr'], data['srclang'])
        
        return success({"status": 200, "translated_list": translated_list})