import numpy as np 
from PIL import Image 
from PIL import ImageFilter
from PIL import ImageEnhance
from PIL import ImageOps
import numpy as np
import matplotlib.pyplot as plt

def convertToYIQ(image):
    yiq = np.array(image, dtype=float)
    yiq[:,:,0] = (image[:,:,0] * 0.299) + (image[:,:,1] * 0.587) + (image[:,:,2] * 0.114)
    yiq[:,:,1] = (image[:,:,0] * 0.596) - (image[:,:,1] * 0.274) - (image[:,:,2] * 0.322)
    yiq[:,:,2] = (image[:,:,0] * 0.211) - (image[:,:,1] * 0.523) + (image[:,:,2] * 0.312)

    return yiq

def negativo_yiq(image):
    yiq = image.copy()
    yiq[:,:,0] = 255 - image[:,:,0]
    return yiq

def negativo(image):
    yiq = image.copy()
    yiq[:,:,0] = 255 - image[:,:,0]
    yiq[:,:,1] = 255 - image[:,:,1]
    yiq[:,:,2] = 255 - image[:,:,2]
    return yiq

def convertToRGB(image):
    rgb = np.array(image, dtype=int)
    r = image[:,:,0] + (image[:,:,1] * 0.956) + (image[:,:,2] * 0.621)
    g = image[:,:,0] - (image[:,:,1] * 0.272) - (image[:,:,2] * 0.647)
    b = image[:,:,0] - (image[:,:,1] * 1.106) + (image[:,:,2] * 1.703)

    rgb[:,:,0] = r
    rgb[:,:,1] = g
    rgb[:,:,2] = b
    rgb = np.clip(rgb, 0, 255)
    return rgb

Shapes = Image.open('apple.png')
print('Mode:',Shapes.mode)
print('Size:',Shapes.size)
print('Type:',type(Shapes))

plt.imshow(Shapes)

ShapesArray = np.array(Shapes)

# img_yiq = negativo(ShapesArray)
img_yiq = convertToYIQ(ShapesArray)
img_yiq = negativo_yiq(img_yiq)
img_yiq = convertToRGB(img_yiq)

plt.imshow(img_yiq, interpolation='nearest')
plt.show()