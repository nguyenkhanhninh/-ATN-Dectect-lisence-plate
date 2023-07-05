import cv2
import pyzbar.pyzbar as pyzbar
import qrcode
from firebase_parking import firebase_parking 
import os
from dotenv import load_dotenv


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
   
    parking_id = os.getenv("parking_id")
    
    if  firebase_parking.get_data_update_in(parking_id) == "open" : 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        barcodes = pyzbar.decode(gray) 
        length = len(barcodes)
        if length == 0:
            firebase_parking.update_barrier_in(parking_id)
        else:
            for barcode in barcodes:
                qr_data = barcode.data.decode("utf-8")
                print(qr_data)
                if qr_data != "close" :
                    firebase_parking.update_barrier_in(parking_id) # Gửi tín hiệu đóng barrier
                    break

    cv2.imshow('QR Code Scanner', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




cap.release()
cv2.destroyAllWindows()