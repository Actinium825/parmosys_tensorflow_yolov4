# Python App for Project Parmosys (Video Object Detection)

## Steps to run
1. Install latest python version
2. Run `python -m venv venv` to create a virtual environment
3. Restart the IDE and make sure `venv` is indicated on the terminal
4. Run `pip install -r requirements.txt`
5. Download my weights and video for detection
   <details>
   <summary>Link</summary>
   
   https://drive.proton.me/urls/ZHDT168A0G#MWuu21jvbTCC

   </details>
6. Place `22class` weights folder in `./checkpoints/22class`
7. Place `test2.wmv` video in `./data/test2.wmv`
8. Skip to Step 10 if running without database
9. Setup database if uploading realtime data:
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
10. Run `python detectvideo.py` or `python detectvideo.py --database {appwrite or firebase}` if running with database
11. To change parking space area, update the `area` flag with other options from [Parmosys Flutter](https://github.com/Actinium825/parmosys_flutter) (snakecase)
    - `python detectvideo.py --database {appwrite or firebase} --area college_of_law`
12. Press Q to exit

## Features
- TensorFlow
- Yolov4
- Realtime Database (Appwrite or Firebase)

## Credits
- Forked from https://github.com/hunglc007/tensorflow-yolov4-tflite

## Screenshot
![](/screenshot.png)
