import os.path
from .OtherSlide import OtherSlide
from .KfbSlide import KfbSlide

from .LRUCacheDict import LRUCacheDict
from threading import Lock
slides = LRUCacheDict()
_dict_lock = Lock()


# public helper: open a slide and return a suitable object
def openSlide(filename):
    ext = os.path.splitext(filename)[1][1:].lower()

    if filename in slides:
        return slides[filename]

    with _dict_lock:
        if filename in slides:
            return slides[filename]

        # print("loading slide: " + filename)

        slide = None
        if ext == 'kfb':# Ningbo Jiangfeng      
            slide = KfbSlide(filename)
        else:# open slide
            slide = OtherSlide(filename)

        slides[filename] = slide
        # print("slide loaded: " + filename)
        return slide

def clearCache():
    slides.clear()