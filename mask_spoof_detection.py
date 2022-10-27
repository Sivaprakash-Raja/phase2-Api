# uvicorn mask_spoof_detection:app --reload 
from pydantic import BaseModel
import uvicorn
import pandas as pd
import numpy as np
import time
import datetime
from fastapi import FastAPI, File, UploadFile
import requests
from openvino.runtime import Core
from typing import List
import cv2
import os
import logging
import traceback

# Function to create directory if its not exists
def create_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)

# log directory
log_path = os.path.join(os.getcwd(), 'logs')
create_folder(log_path)

current_date = str(datetime.datetime.now()).split(' ')[0]
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(filename = os.path.join(log_path,f'mask_spoof_detection_{current_date}.log'),format='%(asctime)s : %(levelname)s : %(message)s')

ie = Core()
app = FastAPI()
# mask detection model
mask_model = ie.read_model(model="models/sbd_mask_classification_224x224.xml", weights="models/sbd_mask_classification_224x224.bin")
mask_compiled_model = ie.compile_model(model=mask_model, device_name="CPU")
# anti_spoof model
anti_spoof_model = ie.read_model(model="models/anti-spoof-mn3.xml", weights="models/anti-spoof-mn3.bin")
anti_spoof_compiled_model = ie.compile_model(model=anti_spoof_model, device_name="CPU")
# face detection model 
face_detection = ie.read_model(model="models/face-detection-retail-0004.xml",weights="models/face-detection-retail-0004.bin")
face_detection_compiled_model = ie.compile_model(model=face_detection, device_name="CPU")

def log_softmax(x):
    e_x = np.exp(x - np.max(x))
    return np.log(e_x / e_x.sum())

class vector(BaseModel):
    is_real : bool
    is_no_mask : bool
    face_detected : bool
    status_message : str

# API endpoint to check status of all conditions
@app.post('/mask_spoof_detection',response_model=vector)
async def mask_spoof_detection(img: UploadFile = File(...)):
     try:
          if '.npy' in img.filename:
               image = np.load(img.file)
          else:
               contents = img.file.read()
               nparr = np.fromstring(contents, np.uint8)
               image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
          logging.info(f'Image succesfully uploaded for the request.')     
          t1_start = time.perf_counter()
          input_image_face_detection = cv2.resize(src=image, dsize=(300, 300))
          input_image_face_detection = np.expand_dims(input_image_face_detection.transpose(2, 0, 1), 0)
          input_key = next(iter(face_detection_compiled_model.inputs))
          output_key = next(iter(face_detection_compiled_model.outputs))
          det_result = face_detection_compiled_model([input_image_face_detection])[output_key]
          xmin_list,xmax_list,ymin_list,ymax_list=[],[],[],[]
          valid_prediction=[]
          for prediction in det_result[0][0]:
               if prediction[2]>0.75:
                    face_detected = True
                    break
               else:
                    face_detected = False
          t1_stop = time.perf_counter()
          logging.info(f"Elapsed time during for face detection program in milliseconds : {float(t1_stop-t1_start)*1000}")
          print("Elapsed time during for face detection program in milliseconds:",float(t1_stop-t1_start)*1000)
          for prediction in det_result[0][0]:
               if prediction[2]>0.75:
                    xmin_list.append(prediction[3])
                    ymin_list.append(prediction[4])
                    xmax_list.append(prediction[5])
                    ymax_list.append(prediction[6])
          if len(xmin_list) != 0 and len(ymin_list) != 0 and len(xmax_list) != 0 and len(ymax_list) != 0:
               rt=[min(xmin_list),min(ymin_list),max(xmax_list),max(ymax_list)]
               x1,y1,x2,y2=int(rt[0]*image.shape[1]),int(rt[1]*image.shape[0]),int(rt[2]*image.shape[1]),int(rt[3]*image.shape[0])
          # print(x1,y1,x2,y2)
               faces = image[y1:y2, x1:x2]
          # print("Faces: ",faces.shape)
          else:
               return {'is_real':False,"is_no_mask":False,"face_detected":False,'status_message':'Show your face'}
          t1_start = time.perf_counter()
          input_image_mask = cv2.resize(src=faces, dsize=(224, 224))
          input_image_mask = np.expand_dims(input_image_mask.transpose(2, 0, 1), 0)
          input_key = next(iter(mask_compiled_model.inputs))
          output_key = next(iter(mask_compiled_model.outputs))
          mask_result = mask_compiled_model([input_image_mask])[output_key]
          index = np.argmax(log_softmax(mask_result))
          if index == 1:
               mask = True
               is_no_mask = False
          else:
               mask = False
               is_no_mask = True
          t1_stop = time.perf_counter()
          print("Elapsed time during for mask detection program in milliseconds:",float(t1_stop-t1_start)*1000)
          logging.info(f"Elapsed time during for mask detection program in milliseconds : {float(t1_stop-t1_start)*1000}")
          t1_start = time.perf_counter()
          input_image_anti_spoof = cv2.resize(src=image, dsize=(128, 128))
          input_image_anti_spoof = np.expand_dims(input_image_anti_spoof.transpose(2, 0, 1), 0)
          input_key = next(iter(anti_spoof_compiled_model.inputs))
          output_key = next(iter(anti_spoof_compiled_model.outputs))
          anti_spoof = anti_spoof_compiled_model([input_image_anti_spoof])[output_key]
          if anti_spoof[0][0] > .125:
               is_real = False
          else:
               is_real = True
          t1_stop = time.perf_counter()
          print("Elapsed time during for anti spoof program in milliseconds:",float(t1_stop-t1_start)*1000)
          logging.info(f"Elapsed time during for anti spoof program in milliseconds : {float(t1_stop-t1_start)*1000}")
          if is_real == True and is_no_mask == True and face_detected == True:  
               return {'is_real':is_real,"is_no_mask":is_no_mask,"face_detected":face_detected,'status_message':'Great job'}
          elif is_real == False and is_no_mask == False and face_detected == True:
               return {'is_real':is_real,"is_no_mask":is_no_mask,"face_detected":face_detected,'status_message':'Show your face'}
          elif is_real == True and is_no_mask == False and face_detected == True:
               return {'is_real':is_real,"is_no_mask":is_no_mask,"face_detected":face_detected,'status_message':'Your face is not visible'}
          elif is_real == False and is_no_mask == True and face_detected == True:
               return {'is_real':is_real,"is_no_mask":is_no_mask,"face_detected":face_detected,'status_message':'Show your face'}
          elif is_real == False and is_no_mask == False and face_detected == False:
               return {'is_real':is_real,"is_no_mask":is_no_mask,"face_detected":face_detected,'status_message':'Show your face'}
     except Exception as e:
          logging.error(f"Error in processing request {traceback.format_exc()}")
          return {'is_real':False, 'is_no_mask':False, 'face_detected':False,'status_message':'Internel server error'}

if __name__=="__main__":
    uvicorn.run("mask_spoof_detection:app",port=50200,log_level="info")  


