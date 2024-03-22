import numpy as np
import cv2

from matplotlib import pyplot as plt

#Lecture image en niveau de gris et conversion en float64
img=np.float64(cv2.imread('../Image_Pairs/FlowerGarden2.png',0))
print(img.shape)
(h,w) = img.shape
print("Dimension de l'image :",h,"lignes x",w,"colonnes")

#Méthode directe
# t1 = cv2.getTickCount()
# img2 = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
# for y in range(1,h-1):
#   for x in range(1,w-1):
#     val = 5*img[y, x] - img[y-1, x] - img[y, x-1] - img[y+1, x] - img[y, x+1] 
#     img2[y,x] = min(max(val,0),255)
# t2 = cv2.getTickCount()
# time = (t2 - t1)/ cv2.getTickFrequency()
# print("Méthode directe :",time,"s")

# cv2.imshow('Avec boucle python',img2.astype(np.uint8))
# #Convention OpenCV : une image de type entier est interprétée dans {0,...,255}
# cv2.waitKey(0)

# plt.subplot(121)
# plt.imshow(img2,cmap = 'gray')
# plt.title('Convolution - Méthode Directe')

# Méthode filter2D
t1 = cv2.getTickCount()
kernel = np.array([[0, -1, 0],[-1, 5, -1],[0, -1, 0]])
img3 = cv2.filter2D(img,-1,kernel)
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Méthode filter2D :",time,"s")

cv2.imshow('Avec filter2D',img3/255.0)
#Convention OpenCV : une image de type flottant est interprétée dans [0,1]
cv2.waitKey(0)

plt.subplot(122)
plt.imshow(img3,cmap = 'gray',vmin = 0.0,vmax = 255.0)
#Convention Matplotlib : par défaut, normalise l'histogramme !
plt.title('Convolution - filter2D')

plt.show()

# Cálculo das componentes do gradiente Ix e Iy usando Sobel
Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

# Cálculo da norma euclidiana do gradiente ||∇I||
grad_norm = np.sqrt(Ix**2 + Iy**2)

# Exibição das imagens
plt.imshow(Ix, cmap='gray')
plt.title('Ix')
plt.show()

plt.imshow(Iy, cmap='gray')
plt.title('Iy')
plt.show()

plt.imshow(grad_norm, cmap='gray')
plt.title('Norme euclidienne du gradient')

plt.show()