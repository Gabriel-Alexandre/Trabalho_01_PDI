import numpy as np 
from PIL import Image 
from PIL import ImageFilter
from PIL import ImageEnhance
from PIL import ImageOps
import numpy as np
import matplotlib.pyplot as plt
import math
import timeit

def media(image, filter, m, n, pivot):
    
    med = image.copy()
    rows = len(image)
    columns = len(image[0])

    pivot_i = pivot[0]-1
    pivot_j = pivot[1]-1
    limit_i = rows-(m-pivot_i)
    limit_j = columns-(n-pivot_j)

    
    if not (m % 2):
        limit_i -= 1

    if not (n % 2):
        limit_j -= 1

    # m2 = m // 2
    # n2 = n // 2
    filter = np.array(filter)
    for i in range(pivot_i, limit_i):
        for j in range(pivot_j, limit_j):
                #med[i][j][0] = np.sum(np.multiply(image[i-m:i+m,j-n:j+n, 0], filter[pivot_i-m2:pivot_i+m2, pivot_j-n2:pivot_j+n2]))
                # med[i][j][1] = np.sum(np.multiply(image[i-m:i+m,j-n:j+n, 1], filter[pivot_i-m2:pivot_i+m2, pivot_j-n2:pivot_j+n2]))
                #med[i][j][2] = np.sum(np.multiply(image[i-m:i+m,j-n:j+n, 2], filter[pivot_i-m2:pivot_i+m2, pivot_j-n2:pivot_j+n2]))

                # input()
                
                med[i][j][0] = abs(round(np.sum(np.multiply(image[i-pivot_i:i+(m-pivot_i),j-pivot_j:j+(n-pivot_j), 0], filter[:,:]))))
                
                med[i][j][1] = abs(round(np.sum(np.multiply(image[i-pivot_i:i+(m-pivot_i),j-pivot_j:j+(n-pivot_j), 1], filter[:,:]))))
                med[i][j][2] = abs(round(np.sum(np.multiply(image[i-pivot_i:i+(m-pivot_i),j-pivot_j:j+(n-pivot_j), 2], filter[:,:]))))

    med = np.clip(med, 0, 255)
    med = med[pivot_i:limit_i, pivot_j:limit_j] 

    return med


aux = open('Q4.txt')
aux = aux.readlines()

offset = float(aux[0])
m, n = aux[2].split()
m = int(m)
n = int(n)

filter = list()
final = 4
for i in range(m):
    filter.append([])
    values = aux[4+i].split()
    for value in values:
        filter[i].append(float(value))
    
    final += 1
final += 1

pivot = aux[final].split()

pivot = [int(pivot[0]), int(pivot[1])]

Shapes = Image.open('apple.png')
Shapes = Shapes.convert('RGB')

print('Mode:',Shapes.mode)
print('Size:',Shapes.size)
print('Type:',type(Shapes))
ShapesArray = np.array(Shapes)

teste = media(ShapesArray, filter, m, n, pivot)

teste= Image.fromarray(teste)
teste = teste.convert('LA')
plt.imshow(teste, interpolation='nearest')
plt.show()





