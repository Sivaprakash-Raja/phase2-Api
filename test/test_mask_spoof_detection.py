import os
import sys
import platform
import cv2
import requests

# change host ip and port number
host_ip = '127.0.0.1'
port = '8000'

if platform.system() == 'Windows':
   abs_path = (("\\").join((os.getcwd()).split("\\")[:-1])).replace('\\','\\\\')
if platform.system() == 'Linux':
   abs_path = ("/").join((os.getcwd()).split("/")[:-1])+"/"
sys.path.insert(0,r'{}'.format(abs_path))




# Test endpoint of mask_spoof_detection
def test_mask_spoof_detection():
   test_image = cv2.imread(os.path.join(abs_path,"image","straight.jpeg"))
   imencoded = cv2.imencode(".jpg", test_image)[1]
   data  ={'img':('image.jpg',imencoded.tobytes(),
                  'image/jpeg',{'Expires': '0'})}
   response = requests.post(f'http://{host_ip}:{port}/mask_spoof_detection',files=data)
   assert response.status_code == 200

def test_mask_model_detection_with_mask_image():
   test_image = cv2.imread(os.path.join(abs_path,"image","with_mask.jpeg"))
   imencoded = cv2.imencode(".jpg", test_image)[1]
   data  ={'img':('image.jpg',imencoded.tobytes(),
                  'image/jpeg',{'Expires': '0'})}
   response = requests.post(f'http://{host_ip}:{port}/mask_spoof_detection',files=data)
   assert response.json()['is_no_mask'] == False  

def test_mask_model_detection_without_mask_image():
   # test_image = cv2.imread(os.path.join(abs_path,"image","WhatsApp Image 2022-09-06 at 2.44.45 PM.jpeg"))
   test_image = cv2.imread(os.path.join(abs_path,"image","straight.jpeg"))
   imencoded = cv2.imencode(".jpg", test_image)[1]
   data  ={'img':('image.jpg',imencoded.tobytes(),
                  'image/jpeg',{'Expires': '0'})}
   response = requests.post(f'http://{host_ip}:{port}/mask_spoof_detection',files=data)
   assert response.json()['is_no_mask'] == True 

def test_spoof_detection():
   test_image = cv2.imread(os.path.join(abs_path,"image","with_mask.jpeg"))
   imencoded = cv2.imencode(".jpg", test_image)[1]
   data  ={'img':('image.jpg',imencoded.tobytes(),
                  'image/jpeg',{'Expires': '0'})}
   response = requests.post(f'http://{host_ip}:{port}/mask_spoof_detection',files=data)
   assert response.json()['is_real'] == True 

def test_spoof_detection_with_fake_image():
   test_image = cv2.imread(os.path.join(abs_path,"image","spoof_image.jpeg"))
   imencoded = cv2.imencode(".jpg", test_image)[1]
   data  ={'img':('image.jpg',imencoded.tobytes(),
                  'image/jpeg',{'Expires': '0'})}
   response = requests.post(f'http://{host_ip}:{port}/mask_spoof_detection',files=data)
   assert response.json()['is_real'] == False 


def test_face_detection():
   test_image = cv2.imread(os.path.join(abs_path,"image","straight.jpeg"))
   imencoded = cv2.imencode(".jpg", test_image)[1]
   data  ={'img':('image.jpg',imencoded.tobytes(),
                  'image/jpeg',{'Expires': '0'})}
   response = requests.post(f'http://{host_ip}:{port}/mask_spoof_detection',files=data)
   assert response.json()['face_detected'] == True    

def test_face_detection_without_face():
   test_image = cv2.imread(os.path.join(abs_path,"image","with_out_face.jpeg"))
   imencoded = cv2.imencode(".jpg", test_image)[1]
   data  ={'img':('image.jpg',imencoded.tobytes(),
                  'image/jpeg',{'Expires': '0'})}
   response = requests.post(f'http://{host_ip}:{port}/mask_spoof_detection',files=data)
   assert response.json()['face_detected'] == False    