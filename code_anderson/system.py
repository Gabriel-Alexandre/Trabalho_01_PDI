import numpy as np 
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import colorsys

class PdiSystem:

    def convertRGBtoYIQ(self, image):
        yiq = np.array(image, dtype=float)
        yiq[:,:,0] = (image[:,:,0] * 0.299) + (image[:,:,1] * 0.587) + (image[:,:,2] * 0.114)
        yiq[:,:,1] = (image[:,:,0] * 0.596) - (image[:,:,1] * 0.274) - (image[:,:,2] * 0.322)
        yiq[:,:,2] = (image[:,:,0] * 0.211) - (image[:,:,1] * 0.523) + (image[:,:,2] * 0.312)

        return yiq
    
    def convertYIQtoRGB(self, image):
        rgb = np.array(image, dtype=float)
        rgb[:,:,0] = image[:,:,0] + (image[:,:,1] * 0.956) + (image[:,:,2] * 0.621)
        rgb[:,:,1] = image[:,:,0] - (image[:,:,1] * 0.272) - (image[:,:,2] * 0.647)
        rgb[:,:,2] = image[:,:,0] - (image[:,:,1] * 1.106) + (image[:,:,2] * 1.703)
        rgb = np.array(rgb, dtype=int)
        rgb = np.clip(rgb, 0, 255)

        return rgb

    def convertToNegativeRGB(self, image):
        negative = np.array(image, dtype=int)
        negative[:,:,0] = 255 - image[:,:,0]
        negative[:,:,1] = 255 - image[:,:,1]
        negative[:,:,2] = 255 - image[:,:,2]
        return negative
    
    def convertToNegativeY(self, image):
        negative = image.copy()
        negative[:,:,0] = 255 - image[:,:,0]
        return negative
    
    @staticmethod
    def read_txt(file):

        aux = open(file)
        aux = aux.readlines()

        offset = float(aux[0])
        m, n = aux[2].split()
        m = int(m)
        n = int(n)

        mask = list()
        final = 4
        for i in range(m):
            mask.append([])
            values = aux[4+i].split()
            for value in values:
                mask[i].append(float(value))
            
            final += 1
        final += 1

        pivot = aux[final].split()

        pivot = [int(pivot[0]), int(pivot[1])]

        return offset, m, n, mask, pivot


    
    def meanFilter(self, file, image):

        offset, m, n, mask, pivot = PdiSystem.read_txt(file)
        
        med = np.array(image, dtype=float)
        rows = len(image)
        columns = len(image[0])

        pivot_i = pivot[0]-1
        pivot_j = pivot[1]-1
        limit_i = rows-(m-pivot_i)
        limit_j = columns-(n-pivot_j)
        

        mask = np.array(mask, dtype=float)
        for i in range(pivot_i, limit_i):
            for j in range(pivot_j, limit_j):    
                    med[i][j][0] = abs(round(np.sum(np.multiply(image[i-pivot_i:i+(m-pivot_i),j-pivot_j:j+(n-pivot_j), 0], mask[:,:])))) + offset
                    med[i][j][1] = abs(round(np.sum(np.multiply(image[i-pivot_i:i+(m-pivot_i),j-pivot_j:j+(n-pivot_j), 1], mask[:,:])))) + offset
                    med[i][j][2] = abs(round(np.sum(np.multiply(image[i-pivot_i:i+(m-pivot_i),j-pivot_j:j+(n-pivot_j), 2], mask[:,:])))) + offset

        med = np.array(med, dtype=int)
        med = np.clip(med, 0, 255)
        med = med[pivot_i:limit_i, pivot_j:limit_j] 

        return med
    
    def medianFilter(self, image, m, n):
        med = image.copy()
        rows = len(image)
        columns = len(image[0])

        row = m // 2
        column = n // 2

        for i in range(row, rows-row):
            for j in range(column, columns-column):
                    med[i][j][0] = np.median(image[i-row:i+row,j-column:j+column, 0])
                    med[i][j][1] = np.median(image[i-row:i+row,j-column:j+column, 1])
                    med[i][j][2] = np.median(image[i-row:i+row,j-column:j+column, 2])

        med = med[row:rows-row, column:columns-column] 
        return med

    def saturationControl(self, s, image):
    
        new_img = np.array(image, dtype=float)
        rows = len(image)
        columns = len(image[0])
        teste2 = new_img.copy()
        for i in range(rows):
            for j in range(columns):
                # print(new_img[i][j])
                new_img[i][j] = colorsys.rgb_to_hsv(image[i][j][0], image[i][j][1], image[i][j][2])
                # print(new_img[i][j])
                #if new_img[i][j][2] < 100:
                 #   new_img[i][j][1] = s
                # print(new_img[i][j])
                # input()
                # if abs(image[i][j][0]+image[i][j][1]-2*image[i][j][2]) > 10:
                # new_img[i][j][1] = s
                # print(new_img[i][j])
                # input()
                #if new_img[i][j][0] != new_img[i][j][1]:
                new_img[i][j][1] = s
            
                teste2[i][j] = colorsys.hsv_to_rgb(new_img[i][j][0], new_img[i][j][1], new_img[i][j][2])
                # print(teste2[i][j])
                # input()
        new_img = np.array(teste2, dtype=int)
        new_img = np.clip(new_img, 0, 255)
    
        return new_img


 

sistema = PdiSystem()

Shapes = Image.open('testpat.1k.color.tif')
print('Mode:',Shapes.mode)
print('Size:',Shapes.size)
print('Type:',type(Shapes))

plt.imshow(Shapes)

ShapesArray = np.array(Shapes)

# file = 'Q3.txt'

# img_yiq = sistema.mean(file, ShapesArray)

# img_yiq = negativo(ShapesArray)
#img_yiq = sistema.convertRGBtoYIQ(ShapesArray)
#img_yiq = sistema.convertToNegativeY(img_yiq)
#img_yiq = sistema.convertYIQtoRGB(img_yiq)
# img_yiq = sistema.medianFilter(ShapesArray, 51, 51)
img_yiq = sistema.saturationControl(0, ShapesArray)
# img_yiq = Image.fromarray(img_yiq)
# img_yiq = img_yiq.convert('L')

plt.imshow(img_yiq)
plt.show()