from flask import Blueprint

m = Blueprint('my_module', __name__)

@m.route('/')
def my_module():
    return 'my_module'