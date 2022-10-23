from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

def Hist(image):
    H=np.zeros(shape=(256,1))
    rows = len(image)
    columns = len(image[0])
    for i in range(rows):
        for j in range(columns):
            k=image[i,j]
            H[k,0]=H[k,0]+1
            
    return H

Shapes = Image.open('DancingInWater.jpg')
imagem = np.array(Shapes)
imagem = np.abs(imagem)

imagem = Image.fromarray(imagem)
Shapes_L = imagem.convert('L')
imagem_F = np.array(Shapes_L)

imagemGrayFinal = Image.fromarray(imagem_F.copy())
Shapes_L1 = imagemGrayFinal.convert('RGB')    
imagem_FF = np.array(Shapes_L1)   

plt.imshow(imagem_FF)
plt.show()

rows = len(imagem_F)
columns = len(imagem_F[0])

histg = Hist(imagem_F)
rmax = np.argmax(histg)
rmin = np.argmin(histg)

rmax = 0
rmin = 255

for i, item in enumerate(histg):
    if item[0] > 0 and i > rmax:
        rmax = i
    if item[0] > 0 and i < rmin:
        rmin = i

imagemGray = imagem_F.copy()
for i in range(rows):
    for j in range(columns):
        if(rmax - rmin) <= 0: k = 0
        else: k = np.abs(np.round(((imagem_F[i, j] - rmin) *255)/(rmax - rmin)))
        imagemGray[i,j] = k

imagemGray = np.abs(imagemGray)
imagemGray = np.clip(imagemGray, 0, 255)
imagemGrayFinal2 = Image.fromarray(imagemGray)
Shapes_L11 = imagemGrayFinal2.convert('RGB')    
imagem_FF2 = np.array(Shapes_L11)   

# print(imagemGray)
plt.imshow(imagem_FF2)
plt.show()