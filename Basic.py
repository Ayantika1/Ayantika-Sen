import cv2
import numpy as np
import face_recognition

imgAyantika = face_recognition.load_image_file('D:/program files/FaceRecognitionProject/ImagesBasic/Ayantika Sen.JPG')
imgAyantika = cv2.cvtColor(imgAyantika,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('D:/program files/FaceRecognitionProject/ImagesBasic/Ayantika Test.JPG')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgAyantika)[0]
encodeAyantika = face_recognition.face_encodings(imgAyantika)[0]
cv2.rectangle(imgAyantika,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(0,0,255),2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(0,0,255),2)

results = face_recognition.compare_faces([encodeAyantika],encodeTest)
faceDis = face_recognition.face_distance([encodeAyantika],encodeTest)
print(results,faceDis)
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('Ayantika Sen',imgAyantika)
cv2.imshow('Ayantika Test',imgTest)
cv2.waitKey(0)