import numpy as np
import cv2

# Se importa la imagen requerida en formato bgr 
img = cv2.imread('road_signs.jpg') # bgr

# Se realiza un suavizado para facilitar el análisis
# img = cv2.GaussianBlur(img, (9,9), 0)
img = cv2.blur(img, (5,5))

# La imagen pasa a formato HSV para realizar una máscara y 
# facilitar la diferencia del color de las señales de tránsito
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
(row, col, ch) = img.shape
# Se define el rango de valores en HSV para la máscara
lower_blue = np.array([15,50,50])
upper_blue = np.array([23,255,163])

# Threshold HSV
mask = cv2.inRange(hsv,lower_blue,upper_blue)

# La función bitwise realiza una comparación de tipo AND de los valores entre la imágen suavizada y la máscara
# y se almacena el resultado en la variable res 
res = cv2.bitwise_and(img,img, mask= mask)

# al final la imagen pasa a escala de grises para encontrar los contornos y poder saber dónde se encuentra la señal de tránsito
res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
canny_img = cv2.Canny(res, 50, 150)
(contornos,_) = cv2.findContours(canny_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img,contornos,-1,(0,0,255), 2)

# Las imágen obtenida se almacena en el computador con el nombre "contornos" para mejor visualización.
cv2.imshow("canny", img)
cv2.imwrite('contornos.png',img)
cv2.waitKey(0)