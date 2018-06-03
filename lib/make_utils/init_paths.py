import sys

import os.path as osp

def add_lib_to_python_path():
    def add_path(path):
        if path not in sys.path:
            sys.path.insert(0, path)
    this_dir = osp.dirname(__file__)
    lib_path = osp.join(this_dir, 'lib')
    add_path(lib_path)
