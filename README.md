# Python App for Project Parmosys (Video Object Detection)

## Steps to run
1. Install latest python version
2. Run `python -m venv venv` to create a virtual environment
3. Restart the IDE and make sure `venv` is indicated on the terminal
4. Run `pip install -r requirements.txt`
5. Place your weights folder in `./checkpoints/{weights}`
6. Place your `.names` file in `./data/classes/{names}`
7. Place your video for detection in `./data/{video}`
8. Setup database if uploading realtime data:
   <details>
   <summary>Appwrite</summary>
   
     1. Create a Project and replace `project_id` in `./data/env.py` with your Project ID
     2. Create an API key with Database scopes enabled and replace `api_key` in `./data/env.py` with your secret key

   </details>
   
   <details>
   <summary>Firebase</summary>
   
     1. Generate a private key `admin_key.json` in your Firebase Project settings' Service accounts tab and place in `./data/{admin_key.json}`
     2. Enable Firestore and create a database
   
   </details>
9. Update `__C.YOLO.CLASSES` directory in `./core/config.py` from step 7
10. Run `python detectvideo.py --weights ./checkpoints/{weights} --video ./data/{video} --database {appwrite or firebase} --area {area}`
11. Press Q to exit

## Features
- TensorFlow
- Yolov4
- Realtime Database (Appwrite or Firebase)

## Screenshot
![](/screenshot.png)
