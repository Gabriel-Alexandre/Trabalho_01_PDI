import numpy as np 
from PIL import Image 
from PIL import ImageFilter
from PIL import ImageEnhance
from PIL import ImageOps
import numpy as np
import matplotlib.pyplot as plt
import math
import timeit

def mediana(image, m, n):
    med = image.copy()
    rows = len(med)
    columns = len(med[0])

    row = m // 2
    column = n // 2

    for i in range(row, rows-row):
        # med[i] = med[i][column:columns-column]
        for j in range(column, columns-column):
                med[i][j][0] = np.median(image[i-row:i+row,j-column:j+column, 0])
                med[i][j][1] = np.median(image[i-row:i+row,j-column:j+column, 1])
                med[i][j][2] = np.median(image[i-row:i+row,j-column:j+column, 2])

    med = med[row:rows-row, column:columns-column] 
    return med


Shapes = Image.open('apple.png')
print('Mode:',Shapes.mode)
print('Size:',Shapes.size)
print('Type:',type(Shapes))
ShapesArray = np.array(Shapes)

tempo = -timeit.default_timer()
teste = mediana(ShapesArray, 19, 19)
tempo += timeit.default_timer()
print('Tempo: ', tempo)

plt.imshow(teste)
plt.show()