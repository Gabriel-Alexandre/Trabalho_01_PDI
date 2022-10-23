import numpy as np 
from PIL import Image 
from PIL import ImageFilter
from PIL import ImageEnhance
from PIL import ImageOps
import numpy as np
import matplotlib.pyplot as plt
import math
import timeit
import colorsys
 
def im_rgb_to_hsb(s, image):
    
    teste = np.array(image, dtype=float)
    # teste = np.array(image)
    rows = len(image)
    columns = len(image[0])
    teste2 = teste.copy()
    for i in range(rows):
        for j in range(columns):
            # teste[i][j] = colorsys.rgb_to_hsv(image[i][j][0]255, image[i][j][1]/255, image[i][j][2]/255)

            teste[i][j] = colorsys.rgb_to_hsv(image[i][j][0], image[i][j][1], image[i][j][2])
            # teste[i][j] = colorsys.rgb_to_hsv(255, 255, 255)
            
            if teste[i][j][0] == teste[i][j][1]:
                pass
            else:
                teste[i][j][1] = s
                
            
                
                # teste[i][j][2] = max(teste[i][j][0]*255, teste[i][j][1]*255, teste[i][j][2])
            # teste[i][j][1] = s
            # print(teste[i][j][1])
            # print(colorsys.hsv_to_rgb(teste[i][j][0], teste[i][j][1], teste[i][j][2]))
            
            teste2[i][j] = colorsys.hsv_to_rgb(teste[i][j][0], teste[i][j][1], teste[i][j][2])
            
            
            # if teste[i][j][1] != 0:
                # print(teste[i][j])
                # input()
            # teste[i][j][0] *= 255
            # teste[i][j][1] *= 255
            # teste[i][j][2] *= 255
            #print(teste[i][j])
            # print(teste[i][j])
            # input()
            
    teste = np.array(teste2, dtype=int)
    
    return teste

Shapes = Image.open('apple.png')
print('Mode:',Shapes.mode)
print('Size:',Shapes.size)
print('Type:',type(Shapes))
ShapesArray = np.array(Shapes)

teste = im_rgb_to_hsb(1, ShapesArray)
plt.imshow(teste)
plt.show()
