from ultralytics import YOLO
import cv2
import cvzone
import math
from PokerHandFunction import findPokerHand

cap = cv2.VideoCapture(0)
# cap.set(3, 800)
# cap.set(4, 600)
#cap = cv2.VideoCapture("../Videos/motorbikes.mp4")


model = YOLO("best.pt")

classNames = ['10C', '10D', '10H', '10S',
              '2C', '2D', '2H', '2S',
              '3C', '3D', '3H', '3S',
              '4C', '4D', '4H', '4S',
              '5C', '5D', '5H', '5S',
              '6C', '6D', '6H', '6S',
              '7C', '7D', '7H', '7S',
              '8C', '8D', '8H', '8S',
              '9C', '9D', '9H', '9S',
              'AC', 'AD', 'AH', 'AS',
              'JC', 'JD', 'JH', 'JS',
              'KC', 'KD', 'KH', 'KS',
              'QC', 'QD', 'QH', 'QS']



while True:
    success , img = cap.read()
    results = model(img, stream=True)
    hand = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            
            # Bounding box
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            #cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            
            w,h = x2-x1,y2-y1
            #bbox = int(x1),int(y1),int(w),int(h)
            cvzone.cornerRect(img, (x1,y1,w,h))
            
            # Confidence
            conf = math.ceil((box.conf[0]*100))/100
            #cvzone.putTextRect(img,f'{conf}', (max(0,x1),max(35,y1-10)))
            
            # Class Name
            cls = int(box.cls[0])
            cvzone.putTextRect(img,f'{classNames[cls]} {conf}', (max(0,x1),max(35,y1-10)), scale=1, thickness=1)
            
            if conf >0.5:
                hand.append(classNames[cls])
    
    print(hand)
    hand = list(set(hand))
    print(hand)
    
    if len(hand) == 5:
        result = findPokerHand(hand)
        print(result)
        cvzone.putTextRect(img,f'Your Hand: {result}', (300,75), scale=3, thickness=5)
            
    cv2.imshow("Image", img)
    cv2.waitKey(1)