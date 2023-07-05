import numpy as nb
import cv2
from pyzbar import pyzbar
import qrcode
import time

# front_cam = cv2.VideoCapture(0)
# back_cam = cv2.VideoCapture(1)

# def scan_qr_code(image):
#     # image = cv2.imread(image_path)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     barcodes = pyzbar.decode(gray)
#     barcode_data = ''
#     for barcode in barcodes:
#         (x, y, w, h) = barcode.rect
#         barcode_data = barcode.data.decode("utf-8")
#         barcode_type = barcode.type

#     return barcode_data

def scan_qr_code(image):
    image = cv2.imread(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    barcodes = pyzbar.decode(gray)
    barcode_data = ''
    for barcode in barcodes:
        (x, y, w, h) = barcode.rect
        barcode_data = barcode.data.decode("utf-8")
        barcode_type = barcode.type

    return barcode_data

#print(scan_qr_code('./anh1.png'))

# def generate_qr_code(data, file_name):
#     qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=10, border=4)
#     qr.add_data(data)
#     qr.make(fit=True)

#     image = qr.make_image(fill_color="black", back_color="white")
#     image.save(file_name)

 
def detect_qr_code(back_cam):

 while True:
    # ret0, frame0 = front_cam.read()
    ret1, frame1 = back_cam.read()

    # if (ret0):
    #     cv2.imshow('cam 0',frame0)
    if (ret1):
        cv2.imshow('cam 1',frame1)
        data = scan_qr_code(frame1)
        print(scan_qr_code(frame1))
        time.sleep(2)
      
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

# detect_qr_code(back_cam)
# front_cam.release()
 back_cam.release()

 cv2.destroyAllWindows()
