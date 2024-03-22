import numpy as np
import cv2
from matplotlib import pyplot as plt

#Lecture image en niveau de gris et conversion en float64
img=np.float64(cv2.imread('../Image_Pairs/Graffiti0.png',cv2.IMREAD_GRAYSCALE))
(h,w) = img.shape
print("Dimension de l'image :",h,"lignes x",w,"colonnes")
print("Type de l'image :",img.dtype)

#Début du calcul
t1 = cv2.getTickCount()
Theta = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)

# Mettre ici le calcul de la fonction d'intérêt de Harris
alpha = 0.06
sizex=7 #fenêtre de sommation de la derivé en x
sgmx= 1#écart-type derivé en x
sizey=7 #fenêtre de sommation de la derivé en y
sgmy=sgmx #écart-type derivé en y
sizem=11 #fenêtre de sommation de la moyenne
sgmm=sgmx*2 #écart-type moyenne

 #Creer noyau de gaussien
gx = (cv2.getGaussianKernel(sizex, sgmx, cv2.CV_64F)).reshape(1,sizex) 
gy = (cv2.getGaussianKernel(sizey, sgmy, cv2.CV_64F))
 #Creer noyau de dérivé de gaussien
x_sg =(-(np.arange(0,sizex)-int(sizex/2))/sgmx**2).reshape(1,sizex)
y_sg =(-(np.arange(0,sizey)-int(sizey/2))/sgmy**2).reshape(sizey,1)
dgx = gx*x_sg
dgy = gy*y_sg
 #Calculer les convolutions
Idx = cv2.filter2D(img,-1,dgx)
Idy = cv2.filter2D(img,-1,dgy)
Idx2= Idx**2
Idy2= Idy**2
Idxy= Idx*Idy
Idx2m = cv2.GaussianBlur(Idx2, (sizem,sizem), sgmm, sgmm)
Idy2m = cv2.GaussianBlur(Idy2, (sizem,sizem), sgmm, sgmm)
Idxym = cv2.GaussianBlur(Idxy, (sizem,sizem), sgmm, sgmm)
Theta = (Idx2m*Idy2m-Idxym**2)-alpha*(Idx2m+Idy2m)**2

# Calcul des maxima locaux et seuillage
Theta_maxloc = cv2.copyMakeBorder(Theta,0,0,0,0,cv2.BORDER_REPLICATE)
d_maxloc = 3
seuil_relatif = 0.1
se = np.ones((d_maxloc,d_maxloc),np.uint8)
Theta_dil = cv2.dilate(Theta,se)
#Suppression des non-maxima-locaux
Theta_maxloc[Theta < Theta_dil] = 0.0
#On néglige également les valeurs trop faibles
Theta_maxloc[Theta < seuil_relatif*Theta.max()] = 0.0
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Mon calcul des points de Harris :",time,"s")
print("Nombre de cycles par pixel :",(t2 - t1)/(h*w),"cpp")

plt.subplot(131)
plt.imshow(img,cmap = 'gray')
plt.title('Image originale')

plt.subplot(132)
plt.imshow(Theta,cmap = 'gray')
plt.title('Fonction de Harris')

se_croix = np.uint8([[1, 0, 0, 0, 1],
[0, 1, 0, 1, 0],[0, 0, 1, 0, 0],
[0, 1, 0, 1, 0],[1, 0, 0, 0, 1]])
Theta_ml_dil = cv2.dilate(Theta_maxloc,se_croix)
#Relecture image pour affichage couleur
Img_pts=cv2.imread('../Image_Pairs/Graffiti0.png',cv2.IMREAD_COLOR)
(h,w,c) = Img_pts.shape
print("Dimension de l'image :",h,"lignes x",w,"colonnes x",c,"canaux")
print("Type de l'image :",Img_pts.dtype)
#On affiche les points (croix) en rouge
Img_pts[Theta_ml_dil > 0] = [255,0,0]
plt.subplot(133)
plt.imshow(Img_pts)
plt.title('Points de Harris')

plt.show()