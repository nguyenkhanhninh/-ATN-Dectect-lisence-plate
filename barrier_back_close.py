import cv2
import pyzbar.pyzbar as pyzbar
import qrcode
from firebase_parking import firebase_parking 
import os
from dotenv import load_dotenv


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    barcodes = pyzbar.decode(gray)
    for barcode in barcodes:
 
        qr_data = barcode.data.decode("utf-8")
        if qr_data == "close":
            parking_id = os.getenv("parking_id")
            firebase_parking.update_barrier_out(parking_id)


    #cv2.imshow('QR Code Scanner', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




cap.release()
cv2.destroyAllWindows()
