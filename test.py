import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

border = 20
imgSize = 300

folder = "Data/L"
counter = 0


labels = ["0","1","2","3","5","7","A", "B", "C", "F", "I", "L"]

while True:
    frame, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgSquare = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgHand = img[y - border:y + h + border, x - border:x + w + border]

        imgHandShape = imgHand.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            
            imgResize = cv2.resize(imgHand, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgSquare[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgSquare, draw=False)
            print(prediction, index)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgHand, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgSquare[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgSquare, draw=False)

        cv2.rectangle(imgOutput, (x - border, y - border-50),
                      (x - border+90, y - border-50+50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x-border, y-border),
                      (x + w+border, y + h+border), (255, 0, 255), 4)


        # cv2.imshow("ImageCrop", imgHand)
        # cv2.imshow("ImageWhite", imgSquare)

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)