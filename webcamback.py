import os
from dotenv import load_dotenv
from datetime import datetime
from PIL import Image
import cv2
import torch
import math 
##from IPython.display import display
import time
import datetime
from datetime import datetime
import argparse
import numpy as np
import math
import random 
import string
import json
from pyzbar import pyzbar
from callAPI import apiPost
import qrcode
from firebase_parking import firebase_parking




# license plate type classification helper function
def linear_equation(x1, y1, x2, y2):
    b = y1 - (y2 - y1) * x1 / (x2 - x1)
    a = (y1 - b) / x1
    return a, b

def check_point_linear(x, y, x1, y1, x2, y2):
    a, b = linear_equation(x1, y1, x2, y2)
    y_pred = a*x+b
    return(math.isclose(y_pred, y, abs_tol = 3))

# detect character and number in license plate
def read_plate(yolo_license_plate, im):
    LP_type = "1"
    results = yolo_license_plate(im)
    bb_list = results.pandas().xyxy[0].values.tolist()
    if len(bb_list) == 0 or len(bb_list) < 7 or len(bb_list) > 10:
        return "unknown"
    center_list = []
    y_mean = 0
    y_sum = 0
    for bb in bb_list:
        x_c = (bb[0]+bb[2])/2
        y_c = (bb[1]+bb[3])/2
        y_sum += y_c
        center_list.append([x_c,y_c,bb[-1]])

    # find 2 point to draw line
    l_point = center_list[0]
    r_point = center_list[0]
    for cp in center_list:
        if cp[0] < l_point[0]:
            l_point = cp
        if cp[0] > r_point[0]:
            r_point = cp
    for ct in center_list:
        if l_point[0] != r_point[0]:
            if (check_point_linear(ct[0], ct[1], l_point[0], l_point[1], r_point[0], r_point[1]) == False):
                LP_type = "2"

    y_mean = int(int(y_sum) / len(bb_list))
    size = results.pandas().s

    # 1 line plates and 2 line plates
    line_1 = []
    line_2 = []
    license_plate = ""
    if LP_type == "2":
        for c in center_list:
            if int(c[1]) > y_mean:
                line_2.append(c)
            else:
                line_1.append(c)
        for l1 in sorted(line_1, key = lambda x: x[0]):
            license_plate += str(l1[2])
        license_plate += "-"
        for l2 in sorted(line_2, key = lambda x: x[0]):
            license_plate += str(l2[2])
    else:
        for l in sorted(center_list, key = lambda x: x[0]):
            license_plate += str(l[2])
    return license_plate

def changeContrast(img):
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl,a,b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_img

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def compute_skew(src_img, center_thres):
    if len(src_img.shape) == 3:
        h, w, _ = src_img.shape
    elif len(src_img.shape) == 2:
        h, w = src_img.shape
    else:
        print('upsupported image type')
    img = cv2.medianBlur(src_img, 3)
    edges = cv2.Canny(img,  threshold1 = 30,  threshold2 = 100, apertureSize = 3, L2gradient = True)
    lines = cv2.HoughLinesP(edges, 1, math.pi/180, 30, minLineLength=w / 1.5, maxLineGap=h/3.0)
    if lines is None:
        return 1

    min_line = 100
    min_line_pos = 0
    for i in range (len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            center_point = [((x1+x2)/2), ((y1+y2)/2)]
            if center_thres == 1:
                if center_point[1] < 7:
                    continue
            if center_point[1] < min_line:
                min_line = center_point[1]
                min_line_pos = i

    angle = 0.0
    nlines = lines.size
    cnt = 0
    for x1, y1, x2, y2 in lines[min_line_pos]:
        ang = np.arctan2(y2 - y1, x2 - x1)
        if math.fabs(ang) <= 30: # excluding extreme rotations
            angle += ang
            cnt += 1
    if cnt == 0:
        return 0.0
    return (angle / cnt)*180/math.pi

def deskew(src_img, change_cons, center_thres):
    if change_cons == 1:
        return rotate_image(src_img, compute_skew(changeContrast(src_img), center_thres))
    else:
        return rotate_image(src_img, compute_skew(src_img, center_thres))



# load model
yolo_LP_detect = torch.hub.load('yolov5', 'custom', path='model/LP_detector_nano_61.pt', force_reload=True, source='local')
yolo_license_plate = torch.hub.load('yolov5', 'custom', path='model/LP_ocr_nano_62.pt', force_reload=True, source='local')
yolo_license_plate.conf = 0.60

prev_frame_time = 0
new_frame_time = 0


back_cam = cv2.VideoCapture(0)

def random_string(len_str):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(len_str))

def scan_qr_code(image):
    # image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    barcodes = pyzbar.decode(gray)
    barcode_data = ''
    for barcode in barcodes:
        (x, y, w, h) = barcode.rect
        barcode_data = barcode.data.decode("utf-8")
        barcode_type = barcode.type
    print(barcode_data)
    return barcode_data

temp_qr_back = ''
lp1 = ""
pre_lp=""
timestamp = datetime.timestamp(datetime.now())

while(True):
   
    ret1, frame1 = back_cam.read()

    
    if ret1:
        plates = yolo_LP_detect(frame1, size=640)
        list_plates = plates.pandas().xyxy[0].values.tolist()
        list_read_plates = set()
        ## lp1 = True
        for plate in list_plates:
            flag = 0
            x = int(plate[0]) # xmin
            y = int(plate[1]) # ymin
            w = int(plate[2] - plate[0]) # xmax - xmin
            h = int(plate[3] - plate[1]) # ymax - ymin  
            crop_img = frame1[y:y+h, x:x+w]
            cv2.rectangle(frame1, (int(plate[0]),int(plate[1])), (int(plate[2]),int(plate[3])), color = (0,0,225), thickness = 2)
        
            
            for cc in range(0,2):
                for ct in range(0,2):
                    lp1 = read_plate(yolo_license_plate, deskew(crop_img, cc, ct))
                    if lp1 != "unknown" and lp1 != "":
                        list_read_plates.add(lp1)
                        cv2.putText(frame1, lp1, (int(plate[0]), int(plate[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                        ##if lp1 =True:
                            
                        flag = 1
                        break
                if flag == 1:
                    break
    
    
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    
    cv2.putText(frame1, str(fps), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
    cv2.imshow('back cam', frame1)
    temp_qr_back = scan_qr_code(frame1)
    key = cv2.waitKey(1)
    if key != -1:
        if key == 113:
            break

    current_time = datetime.today()
    if lp1 != "" and lp1 != "unknown" and lp1 != pre_lp:
        timenow = datetime.timestamp(datetime.now())
        if timenow - timestamp > 1:
            data = {
                "licensePlates": lp1,
                "userId": temp_qr_back,
                "status": "pending",
                }
            dataJson = json.dumps(data)    
            resp =  apiPost.getDetect(dataJson) 
            if resp.status_code == 200:
                resp_json = resp.json()
                timeIn = resp_json["data"]["timeIn"]
                timeIn_format = datetime.strptime(timeIn, '%Y-%m-%dT%H:%M:%SZ')
                timeInstamp = datetime.timestamp(timeIn_format)
                resp1 = apiPost.get_user_information(temp_qr_back)
                if resp1.status_code == 200:
                    respone= resp1.json()
                    parkingHours = (timenow - timeInstamp)/ 3600
                    parkingHours = round(parkingHours)
                    if parkingHours == 0 :
                        parkingHours = 1
                    if respone["data"]["userExists"] == False:
                        token = respone["data"]["token"]
                        id = respone["data"]["paymentId"]
                        url = "http://34.16.145.235:8080/api/auth/payment/paymentViaQR?token={token}&t={parkingHours}&id={id}"
                        url_qr = url.format(token=token, id=id, parkingHours=parkingHours)
                        img = qrcode.make(url_qr)
                        img.save('anh1.png')   # Thanh toán qua QR code khi không đăng kí đinh kì
                        while (True):
                            timeoutformat =current_time.strftime('%Y-%m-%dT%H:%M:%SZ%z')
                            if firebase_parking.firebase_paid(id) == 'paid':
                                data_update = {
                                'timeOut': timeoutformat , 
                                'moneyPaid': 10000* parkingHours,
                                'status': "paid",
                                'parkingId': os.getenv("parking_id"),
                                        }
                        
                                resp_update_detect = apiPost.update_detect(json.dumps(data_update), resp_json["data"]["id"])
                                if resp_update_detect.status_code == 200:
                                    pre_lp = lp1
                                    lp1 = ""
                                    temp_qr_back = ""  
                                    break
                    else :
                        data_payment ={
                            "userId": temp_qr_back,
                            "parkingHours": parkingHours   
                            } 
                        resp_payment = apiPost.request_payment(json.dumps(data_payment), respone["data"]["token"])
                        if resp_payment.status_code == 200:
                            timeoutformat =current_time.strftime('%Y-%m-%dT%H:%M:%SZ%z')
                            data_update = {
                            'timeout':timeoutformat , 
                            'moneyPaid': 10000* parkingHours,
                            'status': "paid",
                            'parkingId': os.getenv("parking_id"),
                                    }
                            ## Thanh toán tự động khi có đăng kí hội viên
                            resp_update_detect = apiPost.update_detect(json.dumps(data_update), resp_json["data"]["id"])
                            if resp_update_detect.status_code == 200:
                                pre_lp = lp1
                                lp1 = ""
                                temp_qr_back = ""
                
    
  
  
    
    # cv2.imwrite('static/uploads/'+random_string(20)+'.png',frame1) # lưu file
    # time.sleep(5) # set thời gian 
         
    
    
    # if lp1 != 'unknown' and lp1 != ''  and temp_qr_back == '': # cam trước detect license plate and create qrcode to save to database
    #     print('licen plate back: ',lp1)
        
            
       
    # if lp1 != 'unknown' and lp1 != '' and temp_qr_back != '' : # detect qr code if have qr and license plate
    #     print ("licen plate back: ", lp1)
    #     print('have qr back: ',temp_qr_back)  
      


back_cam.release()
cv2.destroyAllWindows()