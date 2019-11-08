import os

HERE = os.path.dirname(__file__)
DATA = os.path.abspath(os.path.join(HERE, 'data'))


def _find_resource(filename):
    filename = filename.strip('/')
    return os.path.abspath(os.path.join(DATA, filename))


def get_data(filename):
    return _find_resource(filename)

__all__ = ['get_data']