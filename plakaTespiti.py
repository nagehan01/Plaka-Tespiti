# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 01:11:31 2022

@author: User
"""

import cv2
import numpy as np
import pytesseract as pt
import imutils
pt.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

#resmi import etme
img=cv2.imread('C:\\Users\\User\\Desktop\\plaka_tespiti\\plakaArac.jpg')

#gri tona çevirme
gry=cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

#filtreleme
""" 4 adet değer girilir ilki gri tonlarındaki resimimiz
ikincisi çap
sigma color ve sigma space
"""
filtrelenmis=cv2.bilateralFilter(gry,6,250,250)

# köşeleri algılama
kose=cv2.Canny(filtrelenmis,30,200)

#sınır
"""
cv2.RETR_TREE: konturları daha optimum, güzel şekilde bulmak için kullanılır
cv2.CHAIN_APPROX_SIMPLE: konturlara basit şekilde yaklaşım sağlar.
konturdaki belirsi noktaları gereksiz yerleri yok eder
"""
sinirlar=cv2.findContours(kose,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#imutils kullanarak konturları çekme
snr=imutils.grab_contours(sinirlar)

#sıralama
"""
cv2.contourArea: alana göre sıralama
reverse=True: girilen değeri tersten sıralama
0-10'a kadar
"""
snr=sorted(snr,key=cv2.contourArea,reverse=True)[:10]
ekran=None

# şekil arama
for s in snr:
    epsilon=0.018*cv2.arcLength(s,True)
    yaklasim=cv2.approxPolyDP(s,epsilon,True)
    if len(yaklasim)==4:
        ekran=yaklasim
        break
# maske uygulama
maske=np.zeros(gry.shape,np.uint8)
#plaka bölgesini beyaza çevirme
yeni_img=cv2.drawContours(maske,[ekran],0,(255,255,255),-1)
# yazıyı yapıştırma
yeni_img=cv2.bitwise_and(img,img,mask=maske)

#kırpma işlemi
(x,y)=np.where(maske==(255))
(topx,topy)=(np.min(x),np.min(y))
(bottomx,bottomy)=(np.max(x),np.max(y))
kirp=gry[topx:bottomx+1,topy:bottomy+1]
#okuma
"""
text=pytesseract.image_to_string(kirp,lang="eng")
print(text)"""
text= pt.image_to_string(kirp,config="--psm 11")
print(text)


cv2.imshow("cropted",kirp)

cv2.imshow("orijinal plaka",img)
cv2.imshow("gri plaka",gry)
cv2.imshow("filtrelenmis plaka",filtrelenmis)
cv2.imshow("koselenmis plaka",kose)
#cv2.imshow("maske",maske)
cv2.imshow("yeni maske",yeni_img)
cv2.waitKey(0)
cv2.destroyAllWindows()