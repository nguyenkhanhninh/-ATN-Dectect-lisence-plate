import requests
import asyncio

def getDetect(data):
        url = "http://34.16.145.235:8080/api/detection/get-for-update"
        resp = requests.post(url, data)
        return resp
def get_user_information(id):
        url = ""
        if id == "":
            url = "http://34.16.145.235:8080/api/auth/payment/checkUser"
        else :
            url = "http://34.16.145.235:8080/api/auth/payment/checkUser?id={id}"
            url = url.format(id=id)
        resp = requests.get(url)
        return resp
        

def request_payment(data, token):
        url = "http://34.16.145.235:8080/api/auth/payment/automaticPayment"
        cookies = {'access_token_payment': token}
        resp = requests.post(url, data, cookies=cookies)
        return resp
        
def update_detect(data, id):
        url = "http://34.16.145.235:8080/api/detection/update?id={id}"
        url = url.format(id=id)
        resp = requests.post(url, data)
        return resp




