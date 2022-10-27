## Instruction to setup a virtual environment in your cloud : </br>

python -m virtualenv env </br>
source env/bin/activate </br>
pip install -r requirement.txt </br>


If getting any open-cv error, when depolying in azure VM, try installing this,

sudo apt install libgl1-mesa-glx

-------------------------------------------------------------------------------------- 

## Command to host api :
First activate the virtual environment if not activated </br>
source env/bin/activate </br>

For hosting the Face Vector  endpoint </br>
python face_vector.py

For hosting the Mask Spoof Detection endpoint </br>
python mask_spoof_detection.py



-------------------------------------------------------------------------------------- 

## Test case instructions -- Please host all four endpoints before run this 
source cenv/bin/activate </br>
pip install pytest</br>
Change directory to test</br>
pytest test_face_vector.py</br>
pytest test_mask_spoof_detection.py</br>
