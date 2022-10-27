import cv2
import numpy as np
import requests
import os

def create_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)

temp_path = os.path.join(os.getcwd(), 'temp_test')
create_folder(temp_path)
# Setup camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

pose_list = ['straight', 'left', 'right', 'up', 'down']
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (100,100)
fontScale              = 1
fontColor              = (255,255,255)
thickness              = 1
lineType               = 2

host_ip = "45.126.125.57"
# host_ip = "134.209.145.55"

port = '9010'
user_id = '83743'

print('started')
# While loop
for pose in pose_list:
    nos = 10
    if pose == 'straight' : 
        nos = 50
    for i in range(1,nos+1):
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            print(frame.shape)
            # Show the captured image
            im = frame.copy()
            cv2.putText(im,pose + '_' + str(i), 
                        bottomLeftCornerOfText, 
                        font, 
                        fontScale,
                        fontColor,
                        thickness,
                        lineType)
            cv2.imshow('WebCam', im)
            
            # cv2.imwrite('temp_frame',frame )
            cv2.imwrite(os.path.join(temp_path,"{}_{}.jpeg".format(pose,i)), frame)
            # test_image = cv2.imread(os.path.join(temp_path,"{}_{}.jpeg".format(pose,i)))
            imencoded = cv2.imencode(".jpg", frame)[1]
            data  ={'image':('image.jpg',imencoded.tobytes(),'image/jpeg',{'Expires': '0'})}
            params = {'user_id': user_id,
            'pose': pose,
            'counter': str(i)}
            headers = {'accept': 'application/json'}
            response = requests.post(f'http://{host_ip}:{port}/face_vector_generator',params=params, headers=headers,files=data)
            print(response.json())
            print('Elapsed : ', response.elapsed)

            

            if response.json()['status_message'] == 'Great job' or response.json()['status_message'] == 'Registration complete':
                break


            
            

            
            # wait for the key and come out of the loop
            if cv2.waitKey(1) == ord('q'):
                break

# Discussed below
cap.release()
cv2.destroyAllWindows()
# params = {
#     'user_id': user_id,
# }
# response = requests.post(f'http://{host_ip}:{port}/save_to_database',  params=params, headers=headers )
# print(response.json())


