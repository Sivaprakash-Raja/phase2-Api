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


test_image = cv2.imread(os.path.join(abs_path,"image","WhatsApp Image 2022-09-06 at 3.00.05 PM.jpeg"))
imencoded = cv2.imencode(".jpg", test_image)[1]

# Test endpoint of face_vector_generator
def test_generate_face_vector():
    data  ={'image':('image.jpg',imencoded.tobytes(),'image/jpeg',{'Expires': '0'})}
    params = {'user_id': '12345',
    'pose': 'straight',
    'counter': '1'}
    headers = {'accept': 'application/json'}
    response = requests.post(f'http://{host_ip}:{port}/face_vector_generator',params=params, headers=headers,files=data)
    assert response.status_code == 200

# {"detected_pose":detected_pose,'face_detected':face_detected,'status_message':status}

def test_detect_pose_straight():
   test_image = cv2.imread(os.path.join(abs_path,"image","straight.jpeg"))
   imencoded = cv2.imencode(".jpg", test_image)[1]
   data  ={'image':('image.jpg',imencoded.tobytes(),'image/jpeg',{'Expires': '0'})}
   params = {'user_id': '12345',
   'pose': 'straight',
   'counter': '1'}
   headers = {'accept': 'application/json'}
   response = requests.post(f'http://{host_ip}:{port}/face_vector_generator',params=params, headers=headers,files=data)
   print(response.json())
   assert response.json()['detected_pose'] == 'straight'
   assert response.json()['face_detected'] == True

def test_detect_pose_left():
   test_image = cv2.imread(os.path.join(abs_path,"image","left.jpeg"))
   imencoded = cv2.imencode(".jpg", test_image)[1]
   data  ={'image':('image.jpg',imencoded.tobytes(),'image/jpeg',{'Expires': '0'})}
   params = {'user_id': '12345',
   'pose': 'left',
   'counter': '1'}
   headers = {'accept': 'application/json'}
   response = requests.post(f'http://{host_ip}:{port}/face_vector_generator',params=params, headers=headers,files=data)
   print(response.json())
   assert response.json()['detected_pose'] == 'left'
   assert response.json()['face_detected'] == True

def test_detect_pose_right():
   test_image = cv2.imread(os.path.join(abs_path,"image","right.jpeg"))
   imencoded = cv2.imencode(".jpg", test_image)[1]
   data  ={'image':('image.jpg',imencoded.tobytes(),'image/jpeg',{'Expires': '0'})}
   params = {'user_id': '12345',
   'pose': 'right',
   'counter': '1'}
   headers = {'accept': 'application/json'}
   response = requests.post(f'http://{host_ip}:{port}/face_vector_generator',params=params, headers=headers,files=data)
   print(response.json())
   assert response.json()['detected_pose'] == 'right'
   assert response.json()['face_detected'] == True

def test_detect_pose_up():
   test_image = cv2.imread(os.path.join(abs_path,"image","up.jpeg"))
   imencoded = cv2.imencode(".jpg", test_image)[1]
   data  ={'image':('image.jpg',imencoded.tobytes(),'image/jpeg',{'Expires': '0'})}
   params = {'user_id': '12345',
   'pose': 'up',
   'counter': '1'}
   headers = {'accept': 'application/json'}
   response = requests.post(f'http://{host_ip}:{port}/face_vector_generator',params=params, headers=headers,files=data)
   print(response.json())
   assert response.json()['detected_pose'] == 'up'
   assert response.json()['face_detected'] == True

def test_detect_pose_down():
   test_image = cv2.imread(os.path.join(abs_path,"image","down.jpeg"))
   imencoded = cv2.imencode(".jpg", test_image)[1]
   data  ={'image':('image.jpg',imencoded.tobytes(),'image/jpeg',{'Expires': '0'})}
   params = {'user_id': '12345',
   'pose': 'down',
   'counter': '1'}
   headers = {'accept': 'application/json'}
   response = requests.post(f'http://{host_ip}:{port}/face_vector_generator',params=params, headers=headers,files=data)
   print(response.json())
   assert response.json()['detected_pose'] == 'down'
   assert response.json()['face_detected'] == True

def test_detect_pose_too_left():
   test_image = cv2.imread(os.path.join(abs_path,"image","too_left.jpeg"))
   imencoded = cv2.imencode(".jpg", test_image)[1]
   data  ={'image':('image.jpg',imencoded.tobytes(),'image/jpeg',{'Expires': '0'})}
   params = {'user_id': '12345',
   'pose': 'left',
   'counter': '1'}
   headers = {'accept': 'application/json'}
   response = requests.post(f'http://{host_ip}:{port}/face_vector_generator',params=params, headers=headers,files=data)
   print(response.json())
   assert response.json()['detected_pose'] == 'too left'
   assert response.json()['face_detected'] == True

def test_detect_pose_too_right():
   test_image = cv2.imread(os.path.join(abs_path,"image","too_right.jpeg"))
   imencoded = cv2.imencode(".jpg", test_image)[1]
   data  ={'image':('image.jpg',imencoded.tobytes(),'image/jpeg',{'Expires': '0'})}
   params = {'user_id': '12345',
   'pose': 'right',
   'counter': '1'}
   headers = {'accept': 'application/json'}
   response = requests.post(f'http://{host_ip}:{port}/face_vector_generator',params=params, headers=headers,files=data)
   print(response.json())
   assert response.json()['detected_pose'] == 'too right'
   assert response.json()['face_detected'] == True

def test_detect_pose_too_up():
   test_image = cv2.imread(os.path.join(abs_path,"image","too_up.jpeg"))
   imencoded = cv2.imencode(".jpg", test_image)[1]
   data  ={'image':('image.jpg',imencoded.tobytes(),'image/jpeg',{'Expires': '0'})}
   params = {'user_id': '12345',
   'pose': 'up',
   'counter': '1'}
   headers = {'accept': 'application/json'}
   response = requests.post(f'http://{host_ip}:{port}/face_vector_generator',params=params, headers=headers,files=data)
   print(response.json())
   assert response.json()['detected_pose'] == 'too up'
   assert response.json()['face_detected'] == True

def test_detect_pose_too_down():
   test_image = cv2.imread(os.path.join(abs_path,"image","too_down.jpeg"))
   imencoded = cv2.imencode(".jpg", test_image)[1]
   data  ={'image':('image.jpg',imencoded.tobytes(),'image/jpeg',{'Expires': '0'})}
   params = {'user_id': '12345',
   'pose': 'down',
   'counter': '1'}
   headers = {'accept': 'application/json'}
   response = requests.post(f'http://{host_ip}:{port}/face_vector_generator',params=params, headers=headers,files=data)
   print(response.json())
   assert response.json()['detected_pose'] == 'too down'
   assert response.json()['face_detected'] == True