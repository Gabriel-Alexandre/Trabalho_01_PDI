from system import PdiSystem
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


image = 'einstein.png'

image = Image.open(image)
image = image.convert('RGB')
image = np.array(image)

new = None
L = 256

# RGB-YIQ-RGB
# new = PdiSystem.convertRGBtoYIQ(image)
# new = PdiSystem.convertYIQtoRGB(new)

# Negativo RGB
# new  = PdiSystem.convertToNegativeRGB(image)

# Negativo Y
# new = PdiSystem.convertRGBtoYIQ(image)
# new = PdiSystem.convertToNegativeY(new)
# new = PdiSystem.convertYIQtoRGB(new)

# Correlação (Média)
# file = 'Q3.txt'
# new = PdiSystem.correlationFilter(file, image)

# Correlação (Sobel)
# file = 'Q4.txt'  # Sobel horizontal
# new = PdiSystem.correlationFilter(file, image)

# new_img = Image.fromarray(np.uint8(new))
# new_img = new_img.convert('L')
# new = np.array(new_img)

# plt.imshow(new, cmap='gray')
# plt.show()

# histgram = PdiSystem.histogram(new, L)
# new = PdiSystem.histogramExpansion(new, L, histgram)

file = 'Q5.txt'  # Sobel vertical
new = PdiSystem.correlationFilter(file, image)

new_img = Image.fromarray(np.uint8(new))
new_img = new_img.convert('L')
new = np.array(new_img)

plt.imshow(new, cmap='gray')
plt.show()

histgram = PdiSystem.histogram(new, L)
new = PdiSystem.histogramExpansion(new, L, histgram)

# Mediana
# m = 19
# n = 35
# new = PdiSystem.medianFilter(image, m, n)

# Controle de Saturação
# s = 1
# new = PdiSystem.saturationControl(s, image)


plt.imshow(new, cmap='gray')
# plt.imshow(new)
plt.show()
