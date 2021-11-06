from lib import *

def make_datapath_list(phase="train"):
    rootpath = "./data/"
    target_path = osp.join(rootpath + phase + "/**/*.jpg")
    path_list = []
    for path in glob.glob(target_path):
        path_list.append(path)

    return path_list
