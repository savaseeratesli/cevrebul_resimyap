import cv2

import numpy as np

from random import randint as rnd

camera = cv2.VideoCapture(0)

def nothing(x):
    pass

#Renk bulmak için trackbar
cv2.namedWindow("frame")
cv2.createTrackbar("H1","frame",0,359,nothing)
cv2.createTrackbar("H2","frame",0,359,nothing)
cv2.createTrackbar("S1","frame",0,255,nothing)
cv2.createTrackbar("S2","frame",0,255,nothing)
cv2.createTrackbar("V1","frame",0,255,nothing)
cv2.createTrackbar("V2","frame",0,255,nothing)

kernel=np.ones((7,7),np.uint8)
font=cv2.FONT_HERSHEY_SIMPLEX

#Boş sayfa çizim için
paint=np.ones((480,640,3),np.uint8)*255#3 bgr renk için

#Görüntü ters olmaması için!!!!!!!!!!!!
paint=cv2.flip(paint,1)


while camera.isOpened():
    
    _,frame=camera.read()
    
    frame=cv2.flip(frame,1)
    
    img=frame.copy()
    
    
    #Mavi rengi göster
    
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    
    #Trackbar değerlerini al
    H1=int(cv2.getTrackbarPos("H1","frame")/2) #0-180 arasıdeğer alır
    H2=int(cv2.getTrackbarPos("H1","frame")/2)
    S1=cv2.getTrackbarPos("S1","frame")
    S2=cv2.getTrackbarPos("S2","frame")
    V1=cv2.getTrackbarPos("V1","frame")
    V2=cv2.getTrackbarPos("V2","frame")
    
    #Bulacağamız renk
    lower=np.array([H1,S1,V1])
    upper=np.array([H2,S2,V2])
    
    mask=cv2.inRange(hsv,lower,upper)
    
    #Morfolojik işlemler kasma olmaması için
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
    
    res=cv2.bitwise_and(frame,frame,mask=mask)
 
    
    #Contours renkli cismin çevresini bulma
    contours,_=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #RETR_EXTERNAL
    
    #Şekil çizme cismin alanı
    for i,cnt in enumerate(contours):
        area=cv2.contourArea(cnt)
        if area>5000 or area<200:#Pixel boyutu
            continue
        
        x,y,w,h=cv2.boundingRect(cnt)
        print(x,y,w,h)
           
        color=(rnd(0,256),rnd(0,256),rnd(0,256))#Rastgele renk seçmesi için
        
        M=cv2.moments(cnt)
        center=(int(M['m10']/M['m00']),int(M['m01']/M['m00']))#Kırmızının merkezi bulması için
        
        #Çevre bulma
        perimeter=cv2.arcLength(cnt,True)
        epsilon=0.015*perimeter#ÇEvresini al
        approx=cv2.approxPolyDP(cnt,epsilon,True)
        cv2.drawContours(img,[approx],-1,(0,0,0),5)
        
        #El bulma en uzak çizgileri birleştir
        hull=cv2.convexHull(cnt)
        cv2.drawContours(img,[hull],-1,(255,255,255) ,5)#-1 iç doldurur
        
        
        
        (x2,y2),radius=cv2.minEnclosingCircle(cnt)#İçini boya
        
        center2=(int(x2),int(y2))
        
        #cv2.circle(img,center2,int(radius),(0,255,0),-1)
        
        cv2.circle(img,(x,y),5,(0,0,255),-1)
        
        cv2.circle(img,center,5,(0,0,255),-1)
        
        cv2.drawContours(img,contours,i,color ,4)#-1 iç doldurur
        
        cv2.circle(paint,center2,15,color,-1)
        
        #Köşeye göre cismi bulma kose sayısı bulma
        if len(approx)==3:
            cv2.putText(img,"Ucgen",(x,y),font,1,0,2)
        elif len(approx)==4:
            cv2.putText(img,"Dortgen",(x,y),font,1,0,2)
        elif len(approx)==5:
            cv2.putText(img,"Besgen",(x,y),font,1,0,2)
        elif 6<len(approx)<11:
            cv2.putText(img,"Cokgen",(x,y),font,1,0,2)
        else:
            cv2.putText(img,"Daire",(x,y),font,1,0,2)
            

    
    cv2.imshow("Frame",frame)
   
    cv2.imshow("res",res)
  
    cv2.imshow("img",img)
    
    cv2.imshow("Paint",paint)


    key=cv2.waitKey(5)
    if key==ord("q"):
        break
    elif key==ord("e"):#Paint ekranını yeniler
        paint[:]=255
        

camera.release()
cv2.destroyAllWindows()















