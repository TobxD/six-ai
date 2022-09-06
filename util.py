from hydra.utils import to_absolute_path
import os

def toPath(path):
    if path==None:
        return path
    if path[0] == "/":
        return to_absolute_path(path[1:])
    else:
        return f"{os.getcwd()}/{path}"
