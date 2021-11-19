from OpenGL.raw.GL.VERSION.GL_1_0 import GL_UNPACK_ALIGNMENT, glPixelStorei
from OpenGL.GL import *
import numpy as np
import freetype
import string

import cairosvg
import io
from PIL import Image

def rasterize_svg(*, path: str, width: int, height: int):
    return np.array(Image.open(io.BytesIO(cairosvg.svg2png(url = path, output_width = width, output_height = height))))[:,:,3].astype(np.uint8)

def HSV_to_RGB(H, S, V):

    C = S*V
    X = C*(1 - abs(((H/60)%2) - 1))
    m = V - C

    L = []
    if H < 60:
        L = [C,X,0]
    elif H < 120:
        L = [X,C,0]
    elif H < 180:
        L = [0,C,X]
    elif H < 240:
        L = [0,X,C]
    elif H < 300:
        L = [X,0,C]
    else:
        L = [C,0,X]

    return [L[0] + m, L[1] + m, L[2] + m, 1.0]

def load_font(path = "/home/demoos/Downloads/Metropolis-SemiBold.otf", size = 64*50, dpi = 72):
    font = {}
    face = freetype.Face(path)
    characters = list(string.ascii_lowercase) + \
                list(string.ascii_uppercase) + \
                list(string.digits) + \
                ['č', 'š', 'ž', ':', '-', '#', '.', ',', '!', '?', 0xf008]

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

    face.set_char_size(width=0, height=size, hres=0, vres=dpi)
    for c in characters:
        face.load_char(c)

        bitmap = face.glyph.bitmap
        buffer = np.array(bitmap.buffer, dtype = np.int32).reshape(bitmap.rows, bitmap.width)

        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, buffer.shape[1], buffer.shape[0],
                    0, GL_RED, GL_UNSIGNED_BYTE, buffer)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        font[c] = {"char" : c, 
                   "buffer" : buffer, 
                   "size" : (face.glyph.bitmap.width, face.glyph.bitmap.rows),
                   "bearing" : (face.glyph.bitmap_left, face.glyph.bitmap_top),
                   "advance" : (face.glyph.advance.x, face.glyph.advance.y),
                   "texture" : texture}

    glBindTexture(GL_TEXTURE_2D, 0)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4)
    font["size"] = size     

    return font

def identity4f():
    return np.matrix([[1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0]])

class Parameters():

    def __init__(self, *, font, aspect, state):
        self.font   = font
        self.aspect = aspect 
        self.state  = state