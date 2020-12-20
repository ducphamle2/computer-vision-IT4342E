import random
import string
from json.decoder import JSONDecodeError
from functools import wraps
import hashlib
from datetime import datetime, timedelta

from aiohttp.web import json_response

from errors import ApiBadRequest


def success(response_data, **kwargs):
    return json_response({**kwargs, 'status': 'success', 'data': response_data})


def fail(message, status=400, **kwargs):
    response_data = {'message': message}
    return json_response({**kwargs, 'status': 'fail', 'data': response_data}, status=status)


def store_content_to_file(content, file_path):

    with open(file_path, 'wb') as f:
        f.write((content))


def sha(data):
    if isinstance(data, str):
        data = data.encode('utf-8')
    m = hashlib.sha1()
    m.update(data)
    return m.hexdigest()


def get_request_json(f):
    @wraps(f)
    async def wrapper(*args, **kwargs):
        try:
            request_json = await args[1].json()
            args = (args[0], args[1], request_json)
            return await f(*args, **kwargs)
        except JSONDecodeError:
            raise ApiBadRequest('JSON is required!!!')

    return wrapper


def random_string(stringLength):
    """Generate a random string with the combination of lowercase and uppercase letters """

    letters = string.ascii_letters
    return ''.join(random.choice(letters) for i in range(stringLength))


def get_expiration_date(day):
    dt = datetime.now() + timedelta(days=day)
    return dt