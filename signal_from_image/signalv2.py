import numpy as np
import cv2
from matplotlib import pyplot as plt
import seaborn as sns # Librería usada para generar el estilo de la imagen
sns.set_style('darkgrid')
plt.style.use("ggplot")

# Se importa la imagen de las señales requeridas en formato bgr

imageOut = cv2.imread('signals.png') 

# Se recorta la imagen a la mitad superior para garantizar que se toman las primeras señales
img = imageOut[0:192,0:1260] # Primeras señales
# Se recorta la imagen a la mitad inferior para garantizar que se toman las segundas señales
img2 = imageOut[192:384,0:1260] # Segundas señales

# Se define una función para facilitar la separación de señales
# La función tiene las siguientes características:
# se realiza una conversión en HSV para facilitar el analisis de los valores del color
# La función recoge la imagen seleccionada más un caracter 'b','g' o 'r'. Dependiendo del caracter
# se realiza una máscara siendo que en 'b', por ejemplo, se muestra únicamente la señal azul en un fondo negro
# Seguidamente realiza un barrido de la imagen en Binario, cuando encuentra un valor escribe el valor de la coordenada Y en la que se encuentra el valor
# Solo se recoge el valor de la coordenada. A patir de estos valores se reconstruye la imagen y al final se realiza el escalado necesario para que los valores
# correspondan.

def sign2data(img,color):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    (row, col, ch) = img.shape

    # Se definen los rangos del color en HSV
    if color =='b':
        # Blue
        l = np.array([110,255,255])
        u = np.array([130,255,255])
    elif color == 'g':
        # Green
        l = np.array([50,255,255])
        u = np.array([70,255,255])
    elif color == 'r':
        # Red
        l = np.array([0,255,255])
        u = np.array([30,255,255])

    # Threshold HSV - Máscara de color
    mask = cv2.inRange(hsv,l,u)
    
    # cv2.imshow("signal", mask)
    # cv2.waitKey(0)

# Inicialización de variables para contadores y arreglos de X y Y de las señales
    i=0
    i_x=0    
    x_val=np.zeros(col)
    sign_val=np.zeros(col)

    for i_x in range(col): # in range(min,max,step)
        for i_y in range(row):
            if mask[i_y,i_x] != 0: 
                
                sign_val[i]= i_y
                x_val[i]=i
                                
        if sign_val[i] != 0:
            i=i+1
# Se eliminan los valores innecesarios que quedan en 0                         
    sign_val_v=(sign_val[:i]-107.5)*-1/26.5
    x_val_v=x_val[:i]
  
    return sign_val_v, x_val_v

# imagen 1 --------------------------------------------------------------------
sign_b, x_b=sign2data(img,'b') 
sign_g, x_g=sign2data(img,'g') 
sign_r, x_r=sign2data(img,'r') 
# imagen 2 --------------------------------------------------------------------
sign_b2, x_b2=sign2data(img2,'b') 
sign_g2, x_g2=sign2data(img2,'g')
sign_r2, x_r2=sign2data(img2,'r') 

# Plot señales obtenidas -------------------------------------------------------
fig, axes = plt.subplots(nrows=2, ncols=1)
fig.set_figheight(10)
fig.set_figwidth(50)

axes[0].set_title('Señales obtenidas a partir de Imagen signals.png', fontsize=36)
axes[0].plot(x_b, sign_b,'b',x_g,sign_g,'g',x_r,sign_r,'r')
axes[0].set_xlim(0,1076)
axes[0].set_ylim(-2.9,2.9)
axes[0].set_ylabel('Movement (mm)', fontsize=36)

axes[1].plot(x_b2, sign_b2,'b',x_g2,sign_g2,'g',x_r2,sign_r2,'r')
axes[1].set_ylim(-2.9,2.9)
axes[1].set_xlim(0,1076)
axes[1].set_ylabel('Movement (mm)', fontsize=36)

# Se guarda la figura en una imagen de tipo svg

plt.savefig('fig.svg')