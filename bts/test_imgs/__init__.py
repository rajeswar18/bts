import os
PATH = os.path.dirname(os.path.realpath(__file__))

def get_img(name):
    return os.path.join(PATH, name)

def get_all_imgs():
    return [x for x in os.listdir(PATH) if ".png" in x[-4:] or ".jpg" in x[-4:]]

