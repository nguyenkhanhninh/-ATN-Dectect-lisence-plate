# Import json module
import json

# Khai báo một JSON string
id = '{"user_ID":"647f455ed3f1cad87ec87dc4", "license_plate":"34A-58826", '\
             '"parking_id":"647f467ad3f1cad87ec87dc8", "time_in":"1", "status":""}'

# Đọc JSON String, method này trả về một Dictionary
mylist = json.loads(id)

# In ra thông tin của Dictionary
# print(mylist)

# In ra một giá trị trong Dictionary
print(mylist['user_ID'])
 