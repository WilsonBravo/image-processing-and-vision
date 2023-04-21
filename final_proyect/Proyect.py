import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors


 # ---------------------------------- Vector ----------------------------------**
def Vector (Imagen):
    Imag_HSV = cv2.cvtColor(Imagen,cv2.COLOR_BGR2HSV)
    (row, col, ch) = Imag_HSV.shape
    Imag_V = Imag_HSV[:,:,2]
    
    Row_Vec = np.zeros(row,np.uint8)
    Col_Vec = np.zeros(col,np.uint8)
    Row_Res = np.zeros(row,np.uint8)
    Col_Res = np.zeros(row,np.uint8)
    
    Prom_V = 0
    Max_row = 0
    Max_col = 0
    
    for x in range(row):
        Prom_N = np.mean(Imag_V[x,:])
        Row_Vec[x] = Prom_N
        if Prom_N > Max_row:
            Max_row = Prom_N
            Coord_row = x
            
        
    for y in range(col):
        Prom_N = np.mean(Imag_V[:,y])
        Col_Vec[y] = Prom_N
        if Prom_N > Max_col:
            Max_col = Prom_N
            Coord_col = y
    
    row_M = int(np.divide(row,2))
    col_M = int(np.divide(col,2))
    
    ax = Coord_row-row_M # Coord_row = ax+row_M
    by = Coord_col-col_M 
    

    Angulo = np.arctan2(-ax,by) * 180 / np.pi
    
    Cuadrado1 = np.square(by)
    Cuadrado2 = np.square(ax)
    Magnitud = np.sqrt(Cuadrado1+Cuadrado2)
    
    Vector = np.zeros(6,dtype='float')
    
    Vector[0] = ax
    Vector[1] = by
    
    
    # print('Angulo:', Angulo)
    
    # cv2.imshow('Imagen', Imagen)
    # Imagen_línea = cv2.arrowedLine(Imagen,(col_M, row_M),(Coord_col, Coord_row), (0,0,255),3,cv2.LINE_8,tipLength=0.05)
    # cv2.imshow('Vector1', Imagen_línea)
    # cv2.waitKey(0)
    return Vector

 # ---------------------------------- Count -----------------------------------**
def Count (img):
    original=img
    # Se realiza un suavizado para facilitar el análisis
    # img = cv2.GaussianBlur(img, (9,9), 0)    
    img = cv2.blur(img, (11,11))
    # La imagen pasa a formato HSV para realizar una máscara y 
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    (row, col, ch) = img.shape
    # Se define el rango de valores en HSV para la máscara
    
    l = np.array([0,0,200])
    u = np.array([180,255,255])
    
        
    # Threshold HSV
    mask = cv2.inRange(hsv,l,u)

    # La función bitwise realiza una comparación de tipo AND de los valores entre la imágen suavizada y la máscara
    # y se almacena el resultado en la variable res 
    
    # else:
        
    res_c = cv2.bitwise_and(img,img, mask= mask)

    # al final la imagen pasa a escala de grises para encontrar los contornos y así encontrar el número de objetos dependiendo del color
    res = cv2.cvtColor(res_c, cv2.COLOR_BGR2GRAY)
    canny_img = cv2.Canny(res, 10, 150)
    dilation = cv2.dilate(canny_img, (1,1), iterations=2)
    (contours,_) = cv2.findContours(dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(original,contours,-1,(0,0,255), 2)

    # En la variable contornos se guarda el número de elementos que detecta el algoritmo,
    # A partir de lo siguiente se inserta un texto en la imagen el cual indica el número encontrado.
    # Se tiene en cuenta en ciertos casos un error de conteo debido a que la imagen no tiene formas geometricas perfectas ni un valor de color específico
    # para cada caso, sino un rango de valores y aparte como en el color naranja se encuentran ciertas burbujas y sombras lo cual
    # crea una forma irregular en la cual aunque se implementa un suavizado no es posible de quitar completamente.

    objects = str(len(contours))
    text = "Obj:"+str(objects)
    cv2.putText(canny_img, text, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (240, 0, 159), 1)
    filename ='contornos.jpg'
    # Las imágen obtenida se almacena en el computador con el nombre "contornos" para mejor visualización.
    # cv2.imshow("canny", original)
    # cv2.imshow("Imagen2", res_c)
    #cv2.imwrite(filename,original)
    #cv2.imwrite("contornos.jpg",res_c)
    cv2.waitKey(0)
    
    return res_c

 # ---------------------------------- Centro ----------------------------------**
def Centro_Ilum(Imagen):
    Imag_HSV = cv2.cvtColor(Imagen,cv2.COLOR_BGR2HSV)
    (row, col, ch) = Imag_HSV.shape
    Imag_V = Imag_HSV[:,:,2]
    
    row_M = round(np.divide(row,2))
    col_M = round(np.divide(col,2)) 
    
    suma_Col = np.sum(Imag_V,axis=0)
    suma_Row = np.sum(Imag_V,axis=1)
    
    suma_Total = 0
    suma_Total = np.sum(Imag_V)
    suma_Tx = 0
    suma_Ty = 0
    
    for pos_x in range(col):
        suma_Tx = suma_Tx + (suma_Col[pos_x]*pos_x)
        
    for pos_y in range(row):
        suma_Ty = suma_Ty + (suma_Row[pos_y]*pos_y)
        
    Cent_x = round(np.divide(suma_Tx, suma_Total))
    Cent_y = round(np.divide(suma_Ty, suma_Total))
    Centro = np.zeros(2, dtype = np.uint8)
    Centro = [int(Cent_x)-col_M,int(Cent_y)-row_M]
    
    return Centro


 # --------------------------------- Principal --------------------------------**
def Deteccion_Iluminacion (Image, modo):
    Objetos = []
    Vectores = []
    Contornos = []
    Dimensiones = []
    
    Copia = np.copy(Image)
    Blur_im = cv2.blur((Image), (11,11))
    
    (row, col, ch) = Image.shape
    row_M = round(np.divide(row,2))
    col_M = round(np.divide(col,2))
    
    
    
    # Imagen_General = Count(np.copy(Image))
    # Vector_General = Vector(np.copy(Image))
    # Centro_General = Centro_Ilum(np.copy(Image))
    
 # --------------------------- Modos con los vectores -------------------------**
    if modo == 0: # Solo con el mayor punto de la imágen
        Vector_General = Vector(np.copy(Image))
        
        punto_col = round(col_M+Vector_General[1])
        punto_row = round(row_M+Vector_General[0])
        Image = cv2.arrowedLine(Image,(col_M, row_M),(punto_col, punto_row), (0,0,255),3,cv2.LINE_8,tipLength=0.05)
        
        salx = Vector_General[0]
        saly = Vector_General[1]
        
    elif modo == 1: # Suma del mayor punto con los puntos de los objetos
        Imagen_General = Count(np.copy(Image))
        Vector_General = Vector(np.copy(Image))
        
        Image_g = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
        thresh1 = cv2.Canny(Blur_im, 10, 100)
        thresh2 = cv2.Canny(Image, 1, 255)
    
        thresh = thresh1|thresh2
    
    
        cntr, hrch = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )
    
        hrch = hrch[0]
        temp = 0
        for component in zip(cntr, hrch):
            currentContour = component[0]
            currentHierarchy = component[1]
            x,y,w,h = cv2.boundingRect(currentContour)
        
        
            if currentHierarchy[3] < 0:
                Area = cv2.contourArea(currentContour)
                if  Area>1500 and Area<80000:
                    mask = np.zeros((row, col), np.uint8)
                    cv2.fillPoly(mask, [currentContour], 255)
                
                    mapa = cv2.bitwise_and(np.copy(Copia),np.copy(Copia), mask= mask)
                
                    Objetos.insert(temp, mapa[y:y+h,x:x+w])
                    Vectores.insert(temp, Vector(np.copy(mapa[y:y+h,x:x+w,:])))
                    Dimensiones.insert(temp, [h,w])
                
                    cv2.rectangle(Image,(x,y),(x+w,y+h),(0,0,255),3)
                
                    temp = temp+1
        cv2.putText(Image,"Objetos:"+str(temp), (10,60), cv2.FONT_HERSHEY_PLAIN , 2, (250,0,250), 2)
    
        Vectores = np.array(Vectores)
        Vects_rows = Vectores[:,0]
    
        
        suma_x = int(np.sum(Vectores[:,0])+Vector_General[0])
        suma_y = int(np.sum(Vectores[:,1])+Vector_General[1])
    
        Imagen_línea = cv2.arrowedLine(Image,(col_M, row_M),(col_M+suma_y, row_M+suma_x), (0,0,255),3,cv2.LINE_8,tipLength=0.05)
        
        salx = suma_x
        saly = suma_y
        
    
    
    elif modo == 2:# Suma los mayores puntos de los objetos
        Imagen_General = Count(np.copy(Image))
        
        Image_g = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
        thresh1 = cv2.Canny(Blur_im, 10, 100)
        thresh2 = cv2.Canny(Image, 1, 255)
    
        thresh = thresh1|thresh2
    
    
        cntr, hrch = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )
    
        hrch = hrch[0]
        temp = 0
        for component in zip(cntr, hrch):
            currentContour = component[0]
            currentHierarchy = component[1]
            x,y,w,h = cv2.boundingRect(currentContour)
        
        
            if currentHierarchy[3] < 0:
                Area = cv2.contourArea(currentContour)
                if  Area>1500 and Area<80000:
                    mask = np.zeros((row, col), np.uint8)
                    cv2.fillPoly(mask, [currentContour], 255)
                
                    mapa = cv2.bitwise_and(np.copy(Copia),np.copy(Copia), mask= mask)
                
                    Objetos.insert(temp, mapa[y:y+h,x:x+w])
                    Vectores.insert(temp, Vector(np.copy(mapa[y:y+h,x:x+w,:])))
                    Dimensiones.insert(temp, [h,w])
                
                    cv2.rectangle(Image,(x,y),(x+w,y+h),(0,0,255),3)
                
                    temp = temp+1
                    
            
        cv2.putText(Image,"Objetos:"+str(temp), (10,60), cv2.FONT_HERSHEY_PLAIN , 2, (250,0,250), 2)
    
        Vectores = np.array(Vectores)
        Vects_rows = Vectores[:,0]
        
        suma_x = int(np.sum(Vectores[:,0]))
        suma_y = int(np.sum(Vectores[:,1]))
    
        Imagen_línea = cv2.arrowedLine(Image,(col_M, row_M),(col_M+suma_y, row_M+suma_x), (0,0,255),3,cv2.LINE_8,tipLength=0.05)
        
        salx = suma_x
        saly = suma_y
    
    
    
 # --------------------------- Modos con los Centros --------------------------**
    elif modo == 3: # Centro de iluminación (Basado en densidad lumínica) de toda la imágenes
        Vector_General = Centro_Ilum(np.copy(Image))
        
        punto_col = round(col_M+Vector_General[1])
        punto_row = round(row_M+Vector_General[0])
        Image = cv2.arrowedLine(Image,(col_M, row_M),(punto_col, punto_row), (0,0,255),3,cv2.LINE_8,tipLength=0.05)
        
        salx = Vector_General[0]
        saly = Vector_General[1]
        
    elif modo == 4: # Suma del centro general con los centros de los objetos
        Imagen_General = Count(np.copy(Image))
        Vector_General = Vector(np.copy(Image))
        
        Image_g = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
        thresh1 = cv2.Canny(Blur_im, 10, 100)
        thresh2 = cv2.Canny(Image, 1, 255)
    
        thresh = thresh1|thresh2
    
    
        cntr, hrch = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )
    
        hrch = hrch[0]
        temp = 0
        for component in zip(cntr, hrch):
            currentContour = component[0]
            currentHierarchy = component[1]
            x,y,w,h = cv2.boundingRect(currentContour)
        
        
            if currentHierarchy[3] < 0:
                Area = cv2.contourArea(currentContour)
                if  Area>1500 and Area<80000:
                    mask = np.zeros((row, col), np.uint8)
                    cv2.fillPoly(mask, [currentContour], 255)
                
                    mapa = cv2.bitwise_and(np.copy(Copia),np.copy(Copia), mask= mask)
                
                    Objetos.insert(temp, mapa[y:y+h,x:x+w])
                    Vectores.insert(temp, Centro_Ilum(np.copy(mapa[y:y+h,x:x+w,:])))
                    Dimensiones.insert(temp, [h,w])
                
                    cv2.rectangle(Image,(x,y),(x+w,y+h),(0,0,255),3)
                
                    temp = temp+1
        cv2.putText(Image,"Objetos:"+str(temp), (10,60), cv2.FONT_HERSHEY_PLAIN , 2, (250,0,250), 2)
    
        Vectores = np.array(Vectores)
        Vects_rows = Vectores[:,0]
    
        
        suma_x = int(np.sum(Vectores[:,0])+Vector_General[0])
        suma_y = int(np.sum(Vectores[:,1])+Vector_General[1])
    
        Imagen_línea = cv2.arrowedLine(Image,(col_M, row_M),(col_M+suma_y, row_M+suma_x), (0,0,255),3,cv2.LINE_8,tipLength=0.05)
        
        salx = suma_x
        saly = suma_y
        
    
    elif modo == 5:# Suma los centros de iluminación de los objetos
        Imagen_General = Count(np.copy(Image))
        
        Image_g = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
        thresh1 = cv2.Canny(Blur_im, 10, 100)
        thresh2 = cv2.Canny(Image, 1, 255)
    
        thresh = thresh1|thresh2
    
    
        cntr, hrch = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )
    
        hrch = hrch[0]
        temp = 0
        for component in zip(cntr, hrch):
            currentContour = component[0]
            currentHierarchy = component[1]
            x,y,w,h = cv2.boundingRect(currentContour)
        
        
            if currentHierarchy[3] < 0:
                Area = cv2.contourArea(currentContour)
                if  Area>1500 and Area<80000:
                    mask = np.zeros((row, col), np.uint8)
                    cv2.fillPoly(mask, [currentContour], 255)
                
                    mapa = cv2.bitwise_and(np.copy(Copia),np.copy(Copia), mask= mask)
                
                    Objetos.insert(temp, mapa[y:y+h,x:x+w])
                    Vectores.insert(temp, Vector(np.copy(mapa[y:y+h,x:x+w,:])))
                    Dimensiones.insert(temp, [h,w])
                
                    cv2.rectangle(Image,(x,y),(x+w,y+h),(0,0,255),3)
                
                    temp = temp+1
                    
            
        cv2.putText(Image,"Objetos:"+str(temp), (10,60), cv2.FONT_HERSHEY_PLAIN , 2, (250,0,250), 2)
    
        Vectores = np.array(Vectores)
        Vects_rows = Vectores[:,0]
        
        suma_x = int(np.sum(Vectores[:,0]))
        suma_y = int(np.sum(Vectores[:,1]))
    
        Imagen_línea = cv2.arrowedLine(Image,(col_M, row_M),(col_M+suma_y, row_M+suma_x), (0,0,255),3,cv2.LINE_8,tipLength=0.05)
        
        salx = suma_x
        saly = suma_y
    
    
 # ------------------------ Modos Centros con filtrado ------------------------**
    elif modo == 6: # Centro con filtrado de las zonas menos iluminadas de toda la imágenes
        Imagen_General = Count(np.copy(Image))
        Vector_General = Centro_Ilum(Imagen_General)
        
        punto_col = round(col_M+Vector_General[1])
        punto_row = round(row_M+Vector_General[0])
        Image = cv2.arrowedLine(Image,(col_M, row_M),(punto_col, punto_row), (0,0,255),3,cv2.LINE_8,tipLength=0.05)
        
        salx = Vector_General[0]
        saly = Vector_General[1]
        
    elif modo == 7: # Suma del centro general con los centros de los objetos filtrando
        Imagen_General = Count(np.copy(Image))
        Vector_General = Centro_Ilum(np.copy(Image))
        
        Image_g = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
        thresh1 = cv2.Canny(Blur_im, 10, 100)
        thresh2 = cv2.Canny(Image, 1, 255)
    
        thresh = thresh1|thresh2
    
    
        cntr, hrch = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )
    
        hrch = hrch[0]
        temp = 0
        for component in zip(cntr, hrch):
            currentContour = component[0]
            currentHierarchy = component[1]
            x,y,w,h = cv2.boundingRect(currentContour)
        
        
            if currentHierarchy[3] < 0:
                Area = cv2.contourArea(currentContour)
                if  Area>1500 and Area<80000:
                    mask = np.zeros((row, col), np.uint8)
                    cv2.fillPoly(mask, [currentContour], 255)
                
                    mapa = cv2.bitwise_and(np.copy(Copia),np.copy(Copia), mask= mask)
                
                    Objetos.insert(temp, mapa[y:y+h,x:x+w])
                    Contornos.insert(temp, Count(np.copy(mapa[y:y+h,x:x+w])))
                    Vectores.insert(temp, Centro_Ilum(Contornos[temp]))
                    Dimensiones.insert(temp, [h,w])
                
                    cv2.rectangle(Image,(x,y),(x+w,y+h),(0,0,255),3)
                
                    temp = temp+1
        cv2.putText(Image,"Objetos:"+str(temp), (10,60), cv2.FONT_HERSHEY_PLAIN , 2, (250,0,250), 2)
    
        Vectores = np.array(Vectores)
        Vects_rows = Vectores[:,0]
    
        
        suma_x = int(np.sum(Vectores[:,0])+Vector_General[0])
        suma_y = int(np.sum(Vectores[:,1])+Vector_General[1])
    
        Imagen_línea = cv2.arrowedLine(Image,(col_M, row_M),(col_M+suma_y, row_M+suma_x), (0,0,255),3,cv2.LINE_8,tipLength=0.05)
        
        salx = suma_x
        saly = suma_y
        
    
        # cv2.imshow('Imagen',Image)
        # cv2.waitKey(0)
    
    elif modo == 8:# Suma los centros de iluminación de los objetos con filtrado
        Imagen_General = Count(np.copy(Image))
        
        Image_g = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
        thresh1 = cv2.Canny(Blur_im, 10, 100)
        thresh2 = cv2.Canny(Image, 1, 255)
    
        thresh = thresh1|thresh2
    
    
        cntr, hrch = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )
    
        hrch = hrch[0]
        temp = 0
        for component in zip(cntr, hrch):
            currentContour = component[0]
            currentHierarchy = component[1]
            x,y,w,h = cv2.boundingRect(currentContour)
        
        
            if currentHierarchy[3] < 0:
                Area = cv2.contourArea(currentContour)
                if  Area>1500 and Area<80000:
                    mask = np.zeros((row, col), np.uint8)
                    cv2.fillPoly(mask, [currentContour], 255)
                
                    mapa = cv2.bitwise_and(np.copy(Copia),np.copy(Copia), mask= mask)
                
                    Objetos.insert(temp, mapa[y:y+h,x:x+w])
                    Contornos.insert(temp, Count(np.copy(mapa[y:y+h,x:x+w])))
                    Vectores.insert(temp, Centro_Ilum(Contornos[temp]))
                    Dimensiones.insert(temp, [h,w])
                
                    cv2.rectangle(Image,(x,y),(x+w,y+h),(0,0,255),3)
                
                    temp = temp+1
           
            
        cv2.putText(Image,"Objetos:"+str(temp), (10,60), cv2.FONT_HERSHEY_PLAIN , 2, (250,0,250), 2)
    
        Vectores = np.array(Vectores)
        Vects_rows = Vectores[:,0]
        
        suma_x = int(np.sum(Vectores[:,0]))
        suma_y = int(np.sum(Vectores[:,1]))
    
        Imagen_línea = cv2.arrowedLine(Image,(col_M, row_M),(col_M+suma_y, row_M+suma_x), (0,0,255),3,cv2.LINE_8,tipLength=0.05)
        
        salx = suma_x
        saly = suma_y
        
 # ------------------------ calculos de angulo y fase -------------------------**
   
    
    
    # Poner angulo y magnitud
    Ang = np.arctan2(-salx,saly) * 180 / np.pi
    Mag = np.sqrt(np.square(saly)+np.square(salx))
    Ang = round(Ang,3)
    Mag = round(Mag,3)


    cv2.putText(Image,"Magnitud:"+ str(Mag)+" / Angulo:"+str(Ang), (10,30), cv2.FONT_HERSHEY_PLAIN , 2, (250,50,100), 2)
    
    return Image


 # ----------------------------------- Main -----------------------------------**
# Imag = cv2.imread('Bolas_Brillantes.jpeg')
# Imag = cv2.imread('Escenario.jpeg')
Imag = cv2.imread('Esferas.jpeg')


Modo1 = Deteccion_Iluminacion(np.copy(Imag), 0)
Modo2 = Deteccion_Iluminacion(np.copy(Imag), 1)
Modo3 = Deteccion_Iluminacion(np.copy(Imag), 2)
Modo4 = Deteccion_Iluminacion(np.copy(Imag), 3)
Modo5 = Deteccion_Iluminacion(np.copy(Imag), 4)
Modo6 = Deteccion_Iluminacion(np.copy(Imag), 5)
Modo7 = Deteccion_Iluminacion(np.copy(Imag), 6)
Modo8 = Deteccion_Iluminacion(np.copy(Imag), 7)
Modo9 = Deteccion_Iluminacion(np.copy(Imag), 8)

 # Modos con los vectores
cv2.imshow('Modo 1',Modo1)# Solo el mayor punto de la imagen 
cv2.imshow('Modo 2',Modo2)# Suma el mayor punto de la imagen con los puntos de los objetos
cv2.imshow('Modo 3',Modo3)# Suma los mayores puntos de los objetos

 # Modos con los Centros
cv2.imshow('Modo 4',Modo4)# Centro de iluminación (Basado en densidad lumínica) de toda la imágen
cv2.imshow('Modo 5',Modo5)# Suma del centro general con los centros de los objetos
cv2.imshow('Modo 6',Modo6)# Suma los centros de iluminación de los objetos

 # Modos Centros con filtrado
cv2.imshow('Modo 7',Modo7)# Centro de iluminación (Basado en densidad lumínica) de toda la imágen
cv2.imshow('Modo 8',Modo8)# Suma del centro general con los centros de los objetos
cv2.imshow('Modo 9',Modo9)# Suma los centros de iluminación de los objetos


cv2.waitKey(0)
