import numpy as np 
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import colorsys

class PdiSystem:

    @staticmethod
    def convertRGBtoYIQ(image):
        yiq = np.array(image, dtype=float)
        yiq[:,:,0] = (image[:,:,0] * 0.299) + (image[:,:,1] * 0.587) + (image[:,:,2] * 0.114)
        yiq[:,:,1] = (image[:,:,0] * 0.596) - (image[:,:,1] * 0.274) - (image[:,:,2] * 0.322)
        yiq[:,:,2] = (image[:,:,0] * 0.211) - (image[:,:,1] * 0.523) + (image[:,:,2] * 0.312)

        return yiq
    
    @staticmethod
    def convertYIQtoRGB(image):
        rgb = np.array(image, dtype=float)
        rgb[:,:,0] = image[:,:,0] + (image[:,:,1] * 0.956) + (image[:,:,2] * 0.621)
        rgb[:,:,1] = image[:,:,0] - (image[:,:,1] * 0.272) - (image[:,:,2] * 0.647)
        rgb[:,:,2] = image[:,:,0] - (image[:,:,1] * 1.106) + (image[:,:,2] * 1.703)
        rgb = np.array(rgb, dtype=int)
        rgb = np.clip(rgb, 0, 255)

        return rgb

    @staticmethod
    def convertToNegativeRGB(image):
        negative = np.array(image, dtype=int)
        negative[:,:,0] = 255 - image[:,:,0]
        negative[:,:,1] = 255 - image[:,:,1]
        negative[:,:,2] = 255 - image[:,:,2]
        return negative
    
    @staticmethod
    def convertToNegativeY(image):
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

    @staticmethod    
    def correlationFilter(file, image):

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
    
    @staticmethod
    def medianFilter(image, m, n):
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

    @staticmethod
    def saturationControl(s, image):
    
        new_img = np.array(image, dtype=float)
        rows = len(image)
        columns = len(image[0])
        
        for i in range(rows):
            for j in range(columns):
                
                new_img[i][j] = colorsys.rgb_to_hsv(image[i][j][0], image[i][j][1], image[i][j][2])
                
                new_img[i][j][1] = s
            
                new_img[i][j] = colorsys.hsv_to_rgb(new_img[i][j][0], new_img[i][j][1], new_img[i][j][2])
        new_img = np.array(new_img, dtype=int)
        new_img = np.clip(new_img, 0, 255)
    
        return new_img

    @staticmethod    
    def histogramExpansion(image):
        
        new_img = np.array(image)
        new_img = np.abs(new_img)

        rows = len(new_img)
        columns = len(new_img[0])

        histg = histogram(new_img, 256)

        rmax = 0
        rmin = 255

        for i, item in enumerate(histg):
            if item[0] > 0 and i > rmax:
                rmax = i
            if item[0] > 0 and i < rmin:
                rmin = i

        imagemGray = imagem.copy()
        for i in range(rows):
            for j in range(columns):
                if(rmax - rmin) <= 0: k = 0
                else: k = np.abs(np.round(((imagem[i, j] - rmin) * 255)/(rmax - rmin)))
                imagemGray[i,j] = k
    
    @staticmethod 
    def histogram(image, L):
        H=np.zeros(shape=(L,1))
        rows = len(image)
        columns = len(image[0])
        for i in range(rows):
            for j in range(columns):
                k=image[i,j]
                H[k,0]=H[k,0]+1
                
        return H

