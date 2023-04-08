from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture(1 , cv2.CAP_DSHOW)
cap.set(3, 1280)
cap.set(4, 720)
#cap = cv2.VideoCapture("../Videos/ppe-1.mp4")


model = YOLO("best.pt")

classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest',
              'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']

myColor = (255,0,0)
txtColor = (255,255,255)


while True:
    success , img = cap.read()
    results = model(img, stream=True)
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            
            # Bounding box
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            #cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            
            w,h = x2-x1,y2-y1
            #bbox = int(x1),int(y1),int(w),int(h)
            #cvzone.cornerRect(img, (x1,y1,w,h))
            
            # Confidence
            conf = math.ceil((box.conf[0]*100))/100
            #cvzone.putTextRect(img,f'{conf}', (max(0,x1),max(35,y1-10)))
            
            # Class Name
            cls = int(box.cls[0])
            
            currentClass = classNames[cls]
            
            if conf > 0.5:
                if currentClass == 'Hardhat' or currentClass == 'Mask' or currentClass == 'Safety Vest':
                    myColor = (0,255,0)
                    txtColor = (0,0,0)
                elif currentClass == 'NO-Hardhat' or currentClass == 'NO-Mask' or currentClass == 'NO-Safety Vest':
                    myColor = (0,0,255)
                    txtColor = (255,255,255)
                else:
                    myColor = (255,0,0)
                    txtColor = (255,255,255)
                
                cvzone.putTextRect(img,f'{currentClass} {conf}', 
                                (max(0,x1),max(35,y1-5)), scale=1, thickness=1,
                                colorB=myColor,colorT=txtColor,colorR=myColor, offset=5)
                
                cv2.rectangle(img,(x1,y1),(x2,y2), myColor,3)
            
    cv2.imshow("Image", img)
    cv2.waitKey(1)