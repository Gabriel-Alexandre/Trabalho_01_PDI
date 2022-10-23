import numpy as np 
from PIL import Image 
from PIL import ImageFilter
from PIL import ImageEnhance
from PIL import ImageOps
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import cm


def convertToYIQ(image):
    yiq = np.array(image, dtype=float)
    yiq[:,:,0] = (image[:,:,0] * 0.299) + (image[:,:,1] * 0.587) + (image[:,:,2] * 0.114)
    yiq[:,:,1] = (image[:,:,0] * 0.596) - (image[:,:,1] * 0.274) - (image[:,:,2] * 0.322)
    yiq[:,:,2] = (image[:,:,0] * 0.211) - (image[:,:,1] * 0.523) + (image[:,:,2] * 0.312)

    return yiq

def convertToRGB(image):
    rgb = np.array(image, dtype=int)
    rgb[:,:,0] = image[:,:,0] + (image[:,:,1] * 0.956) + (image[:,:,2] * 0.621)
    rgb[:,:,1] = image[:,:,0] - (image[:,:,1] * 0.272) - (image[:,:,2] * 0.647)
    rgb[:,:,2] = image[:,:,0] - (image[:,:,1] * 1.106) + (image[:,:,2] * 1.703)
  
    return rgb


Shapes = Image.open('DancingInWater.jpg')
print('Mode:',Shapes.mode)
print('Size:',Shapes.size)
print('Type:',type(Shapes))

plt.imshow(Shapes)

ShapesArray = np.array(Shapes)

imageYIQ = convertToYIQ(ShapesArray)

imageRGB = convertToRGB(imageYIQ)
plt.imshow(imageRGB, interpolation='nearest')
plt.show()

# plt.plot()

#print(plt.imshow(imageRGB))