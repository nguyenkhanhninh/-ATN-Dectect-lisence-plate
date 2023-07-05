import firebase_admin
from firebase_admin import credentials
from firebase_admin import db


def firebase_paid(id):
    # Khởi tạo ứng dụng Firebase
    if not firebase_admin._apps:
        cred = credentials.Certificate('firebase_parking\smart-parking-firebase-adminsdk.json')
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://smart-parking-b4df2-default-rtdb.firebaseio.com/'
        })
    
    # Xây dựng đường dẫn truy vấn trong cơ sở dữ liệu Firebase
    url = '/payment/{id}'
    url = url.format(id=id)
  
    # Lấy tham chiếu đến nút dữ liệu cần truy vấn
    ref = db.reference(url)
  
    # Lấy dữ liệu từ Firebase
    data = ref.get()
  
    # In dữ liệu
    print(data)

    return data


def get_data_update_in(parkingId):
    if not firebase_admin._apps:
        cred = credentials.Certificate('firebase_parking\smart-parking-firebase-adminsdk.json')
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://smart-parking-b4df2-default-rtdb.firebaseio.com/'
        })
    path = '{parkingId}/barrier_in'
    path = path.format(parkingId=parkingId)
    ref = db.reference(path)
    data = ref.get()
    return data

def get_data_update_out(parkingId):
    if not firebase_admin._apps:
        cred = credentials.Certificate('firebase_parking\smart-parking-firebase-adminsdk.json')
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://smart-parking-b4df2-default-rtdb.firebaseio.com/'
        })
    path = '{parkingId}/barrier_out'
    path = path.format(parkingId=parkingId)
    ref = db.reference(path)
    data = ref.get()
    return data

def update_barrier_out(parkingId):
    if not firebase_admin._apps:
        cred = credentials.Certificate('firebase_parking\smart-parking-firebase-adminsdk.json')
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://smart-parking-b4df2-default-rtdb.firebaseio.com/'
        })
    
 
    path = '{parkingId}/barrier_out'
    path = path.format(parkingId=parkingId)
      
      # Cập nhật giá trị của node
    ref = db.reference(path)
    ref.set('close')


def update_barrier_in(parkingId):
    if not firebase_admin._apps:
        cred = credentials.Certificate('firebase_parking\smart-parking-firebase-adminsdk.json')
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://smart-parking-b4df2-default-rtdb.firebaseio.com/'
        })
    
     # Tạo đường dẫn đến node cần cập nhật
    path = '{parkingId}/barrier_in'
    path = path.format(parkingId=parkingId)
      
      # Cập nhật giá trị của node
    ref = db.reference(path)
    ref.set('close')
