from flask_restful.representations import json

class InvalidParameter(APIException):
   status_code = 204
   detail = 'Invalid parameters'