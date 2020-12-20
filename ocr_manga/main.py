import logging
import time

from aiohttp import web
import aiohttp_cors

from handler import route_handler
from includes.utils import *
from config import *

LOGGER = logging.getLogger(__name__)

app = web.Application(client_max_size=Config.CLIENT_MAX_SIZE)

handler = route_handler.RouteHandler()

app.router.add_post('/v1/translate', handler.translate)


cors = aiohttp_cors.setup(app, defaults={
    "*": aiohttp_cors.ResourceOptions(
        allow_credentials=True,
        expose_headers="*",
        allow_headers="*",
    )
})


for route in list(app.router.routes()):

    if not isinstance(route.resource, web.StaticResource):  # <<< WORKAROUND
        cors.add(route)

web.run_app(app, host=Config.HOST , port=Config.PORT)
