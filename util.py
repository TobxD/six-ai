from hydra.utils import to_absolute_path
import os

def toPath(path):
    if path==None:
        return path
    if path[0] == "/":
        return f"{os.getcwd()}/../../..{path}"
    else:
        return f"{os.getcwd()}/{path}"
