# uvicorn face_vector:app --reload 
from pydantic import BaseModel
import uvicorn
import numpy as np
import time
import datetime
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
import requests
from openvino.runtime import Core
import cv2
import psycopg2
from config import *
import os
import logging
import traceback
import platform
import sys
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
import base64
import shutil


if platform.system() == 'Windows':
   abs_path = (("\\").join((os.getcwd()).split("\\")[:])).replace('\\','\\\\')
if platform.system() == 'Linux':
   abs_path = ("/").join((os.getcwd()).split("/")[:])+"/"
sys.path.insert(0,r'{}'.format(abs_path))
# Function to create directory if its not exists
def create_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)

# log directory
log_path = os.path.join(os.getcwd(), 'logs')
temp_path = os.path.join(os.getcwd(), 'temp_data')

create_folder(log_path)
create_folder(temp_path)


current_date = str(datetime.datetime.now()).split(' ')[0]
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(filename = os.path.join(log_path,f'face_vector_generate_{current_date}.log'),format='%(asctime)s : %(levelname)s : %(message)s')



ie = Core()
app = FastAPI()

# face detection model 
face_detection = ie.read_model(model=os.path.join(abs_path,"models","face-detection-retail-0004.xml"),weights=os.path.join(abs_path,"models","face-detection-retail-0004.bin"))
face_detection_compiled_model = ie.compile_model(model=face_detection, device_name="CPU")
# feature embedding model
feature_embedding = ie.read_model(model=os.path.join(abs_path,"models","face-recognition-arcface-112x112.xml"),weights=os.path.join(abs_path,"models","face-recognition-arcface-112x112.bin"))
feature_embedding_compiled_model = ie.compile_model(model=feature_embedding, device_name="CPU")
# head pose detection model
head_pose_model = ie.read_model(model=os.path.join(abs_path,"models","head-pose-estimation-adas-0001.xml"), weights=os.path.join(abs_path,"models","head-pose-estimation-adas-0001.bin"))
head_pose_compiled_model = ie.compile_model(model=head_pose_model, device_name="CPU")

def decode_pose(yaw, pitch, roll):
    vals = np.array([abs(pitch),abs(yaw),abs(roll)])
    max_index = np.argmax(vals)
    MIN_THRESHOLD = 5
    if vals[max_index] <= MIN_THRESHOLD: 
        txt = "straight"

    else:
        if max_index == 0:
            if abs(pitch) < 20:
                if pitch > 0: 
                        txt = "down"
                else: 
                        txt = "up"
            else: 
                if pitch > 0: 
                        txt = "too down"
                else: 
                        txt = "too up"
        elif max_index == 1:
            if abs(yaw) < 20:
                if yaw > 0: 
                        txt = "left" 
                else: 
                        txt = "right"
            else: 
                if yaw > 0: 
                        txt = "too left"
                else: 
                        txt = "too right"
        elif max_index == 2:
            if roll > 0: txt = "tilting left"
            else: txt = "tilting right"
        
    return txt

# encrypt function is to encrypt image for security
def encrypt(image):
    try:
        image = bytes(image, 'utf-8')
        cipher = AES.new(key, AES.MODE_ECB)
        encrypted_image = cipher.encrypt(pad(image, AES.block_size))
        encrypted_image_string = str(base64.b64encode(encrypted_image),'utf-8')
        return encrypted_image_string
    except:
        logging.error(traceback.format_exc())
        return False

# function to get user id and save face vector to db and change regr status and save images to db
def save_images_to_db(product_id):
    # get actual user_id
    try:
        conn = psycopg2.connect(database=database, user=user, password=password, host=host, port=port)
        cursor = conn.cursor()
        query_uid = f'select user_id from {schema_name}.{register_svp_table_name} where product_id=%s'        
        cursor.execute(query_uid, (product_id,))
        user_id = cursor.fetchall()[0][0]
        cursor.close()
        conn.close()
    except:
        logging.error(f'Error in getting userid from product user table')
        return
    # save face vectors to database
    save_fv_to_database(product_id = product_id,user_id = user_id)
    # update user active status
    post_data = {'user_id':user_id,'device_id' : user_id}
    try:
        response = requests.post(update_status_url, json=post_data)
        update_status = response.json()['status']
        if update_status == 'updated':
            logging.info(f'product_id - {product_id} status is updated successfully in product_users table')
        if 'Error' in update_status:
            logging.error(f'product_id - {product_id} {str(update_status).lower()}')
    except Exception as e:
        if 'HTTPConnectionPool' in str(e):
            logging.error(f'please check hosted url - {update_status_url} configuration')
        else:
            logging.error(f"Error in updating status for {product_id}:\n {traceback.format_exc()}")
    # save images to database
    try:
        logging.info(f'save images to database started for product_id : {product_id}')
        temp_user_path_image = os.path.join(temp_path, product_id, 'image')
        image_list = os.listdir(temp_user_path_image)
        # database connection is established
        conn = psycopg2.connect(database=database, user=user, password=password, host=host, port=port)
        cursor = conn.cursor()
        for img in image_list:
            image = np.load(os.path.join(temp_user_path_image , img))
            image_data = encrypt(str(image.tolist()))
            value_list = img.split('__ford__')
            # user_id = value_list[0]
            frame_number = value_list[1]
            instruction = value_list[2]
            extension = value_list[-1].split('.')[-1]
            time_stamp = f"{value_list[3]} {value_list[4]}:{value_list[5]}:{(value_list[-1]).split('.')[0]}"
            image_name = f"{user_id}_{frame_number}_{instruction}_{((time_stamp).replace(' ','_')).replace(':','_')}.{extension}"
            # data is inserted to database
            cursor.execute(f"INSERT INTO {schema_name}.{register_image_table_name} (timestamp,user_id,image_name,image) VALUES('{time_stamp}','{user_id}','{image_name}','{image_data}')")
            conn.commit()
            # if os.path.exists(os.path.join(temp_user_path_image , img)):
            #     os.remove(os.path.join(temp_user_path_image , img))
        logging.info(f"Image insertion completed for product_id:{product_id} in {register_image_table_name}")
        cursor.close()
        conn.close()
        shutil.rmtree(os.path.join(temp_path, product_id))
        logging.info(f"Deleted data in temp folder for product_id:{product_id}")
    except:
        logging.error(f"Error in uploading image for {product_id}:\n {traceback.format_exc()}")


# def save_to_database(user_id: str, background_task: BackgroundTasks):
def save_fv_to_database( product_id: str, user_id: str,):
    logging.info(f'save face vector to database requested for product_id : {product_id}')

    try:
        temp_user_path_face = os.path.join(temp_path, product_id, 'face')
        image_list = os.listdir(temp_user_path_face)
    except:
        logging.error(f"Error in processing request for {product_id}:\n {traceback.format_exc()}")
        return {'status': 'No temp data'}
    try:
        # database connection is established
        conn = psycopg2.connect(database=database, user=user, password=password, host=host, port=port)
        cursor = conn.cursor()
        for img in image_list:
            # t1_start = time.perf_counter()
            faces = np.load(os.path.join(temp_user_path_face , img))

            image_input = cv2.resize(src=faces, dsize=(112, 112))
            image_input=np.expand_dims(np.moveaxis(image_input, 2 , 0),0)
            input_key_feature = next(iter(feature_embedding_compiled_model.inputs))
            output_key_feature = next(iter(feature_embedding_compiled_model.outputs))
            vec_result = feature_embedding_compiled_model([image_input])[output_key_feature]
            face_vector = vec_result[0]
            face_vector = encrypt(str(face_vector.tolist()))

            value_list = img.split('__ford__')

            frame_number = value_list[1]
            instruction = value_list[2]
            extension = value_list[-1].split('.')[-1]
            time_stamp = f"{value_list[3]} {value_list[4]}:{value_list[5]}:{(value_list[-1]).split('.')[0]}"
            image_name = f"{user_id}_{frame_number}_{instruction}_{((time_stamp).replace(' ','_')).replace(':','_')}.{extension}"
            
            
            # data is inserted to database
            cursor.execute(f"INSERT INTO {schema_name}.{register_table_name} (user_id,image_name,face_vector,timestamp) VALUES('{user_id}','{image_name}','{face_vector}','{time_stamp}')")
            conn.commit()
            

        cursor.close()
        conn.close()
        logging.info(f"face vector generation and insertion completed for product_id:{product_id} in {register_table_name} ")
        # background_task.add_task(save_images_to_db, user_id)
    except:
        logging.error(f"Error in fv generation for {product_id}:\n {traceback.format_exc()}")
        return {'status': 'Internal server error'}
    return {'status': 'ok'}


class FaceVector(BaseModel):
    detected_pose : str
    face_detected : bool
    status_message : str

# API endpoint to check head pose and save image
@app.post("/face_vector_generator",response_model=FaceVector)
def generate_face_vector(user_id: str ,pose:str,counter: int,background_task: BackgroundTasks,image :UploadFile = File(...)):
    # check if registration is just staring if temp folder already exist remove 
    if pose == 'straight' and counter == 1 and  os.path.exists(os.path.join(temp_path, user_id,'image')):
        if len(os.listdir(os.path.join(temp_path, user_id, 'image')))>1:
            shutil.rmtree(os.path.join(temp_path, user_id))
    # create folders for images and cropped faces
    temp_user_path_image = os.path.join(temp_path, user_id, 'image')
    temp_user_path_face = os.path.join(temp_path, user_id, 'face')
    
    create_folder(os.path.join(temp_path, user_id))
    create_folder(temp_user_path_image)
    create_folder(temp_user_path_face)

    try:
        # Read input image
        detected_pose = ''
        face_detected = ''
        if '.npy' in image.filename:
            image = np.load(image.file)
        else:
            contents = image.file.read()
            nparr = np.fromstring(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # Face detection model inference
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
        for prediction in det_result[0][0]:
            if prediction[2]>0.75:
                xmin_list.append(prediction[3])
                ymin_list.append(prediction[4])
                xmax_list.append(prediction[5])
                ymax_list.append(prediction[6])
        if len(xmin_list) != 0 and len(ymin_list) != 0 and len(xmax_list) != 0 and len(ymax_list) != 0:
            # print("list values of list:\n",len(xmin_list),len(ymin_list),len(xmax_list),len(ymax_list))
            rt=[min(xmin_list),min(ymin_list),max(xmax_list),max(ymax_list)]
            
        else:
            return {"detected_pose":'nil','face_detected':face_detected,'status_message':'Show your face'}
        
        # print(rt)
        x1,y1,x2,y2=int(rt[0]*image.shape[1]),int(rt[1]*image.shape[0]),int(rt[2]*image.shape[1]),int(rt[3]*image.shape[0])
        faces = image[y1:y2, x1:x2]
        # print("Faces: ",faces.shape)
        t1_stop = time.perf_counter()
        print("Elapsed time during for face detection program in milliseconds:",float(t1_stop-t1_start)*1000)
        logging.info(f"Elapsed time during for face detection program in milliseconds : {float(t1_stop-t1_start)*1000}")
        height = image.shape[0]
        width = image.shape[1]
        x1_norm = x1 - (0.03*width)
        y1_norm = y1 - (0.03*height)
        x2_norm = x2 + (0.03*width)
        y2_norm = y2 + (0.03*height)
        faces_norm = image[int(y1_norm):int(y2_norm), int(x1_norm):int(x2_norm)]
        # print(faces_norm.shape)
        t1_start = time.perf_counter()
        input_image_head_pose = cv2.resize(src=faces_norm, dsize=(60, 60))
        input_image_head_pose = np.expand_dims(input_image_head_pose.transpose(2, 0, 1), 0)
        input_key = next(iter(head_pose_compiled_model.inputs))
        output_key = next(iter(head_pose_compiled_model.outputs))        
        head_pose_result = head_pose_compiled_model([input_image_head_pose])
        list_of_values = list(head_pose_result.values())
        roll  = list_of_values[0][0][0]
        pitch = list_of_values[1][0][0]
        yaw   = list_of_values[2][0][0]
        detected_pose = decode_pose(yaw,pitch,roll)
        t1_stop = time.perf_counter()
        print("Elapsed time during for head pose program in milliseconds:",float(t1_stop-t1_start)*1000)
        logging.info(f"Elapsed time during for head pose program in milliseconds : {float(t1_stop-t1_start)*1000}")    
        if pose == detected_pose and face_detected == True:
            time_stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            spliter = "__ford__"
            image_name_local = f"{user_id}{spliter}{str(counter)}{spliter}{pose}{spliter}{(str(time_stamp).replace(' ',spliter).replace(':',spliter))}.npy"
            image_path_img = os.path.join(temp_user_path_image, image_name_local)
            np.save(image_path_img,image)

            image_path_face = os.path.join(temp_user_path_face, image_name_local)
            np.save(image_path_face,faces)

            if pose == 'down' and counter == 10 and os.path.exists(os.path.join(temp_path, user_id,'image')):
                if len(os.listdir(os.path.join(temp_path, user_id, 'image'))) > 0:
                    background_task.add_task(save_images_to_db, user_id)
                    return {"detected_pose":detected_pose,'face_detected':face_detected,'status_message':'Registration complete'}

            return {"detected_pose":detected_pose,'face_detected':face_detected,'status_message':'Great job'}
        elif  pose != detected_pose and face_detected == True:
            logging.info(f"Issue in one of the model for the current image in the request with product_id:{user_id}")
            if detected_pose == 'out_of_view':
                status = f'{detected_pose}'
            else:
                status = f'You are looking {detected_pose}'
            return {"detected_pose":detected_pose,'face_detected':face_detected,'status_message':status}
    except Exception as e:
        logging.error(f"Error in processing request for {user_id}:\n {traceback.format_exc()}")
        # print(traceback.format_exc())
        return {"detected_pose":'nil','face_detected':False,'status_message':'Internel server error'}
    

if __name__=="__main__":
    uvicorn.run("face_vector:app",port=50100,log_level="info")
 
