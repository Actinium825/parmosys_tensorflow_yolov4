# Python App for Project Parmosys (Video Object Detection)

## Steps to run
1. Install latest python version
2. Run `python -m venv venv` to create a virtual environment
3. Choose the created virtual environment as the python interpreter for the project
4. Restart the IDE and make sure `venv` is indicated on the terminal
5. Run `pip install -r requirements.txt`
6. Place your weights folder in `./checkpoints/{weights}`
7. Place your `.names` file in `./data/classes/{names}`
8. Place your video for detection in `./data/{video}`
9. If using Appwrite, add `env.py` to `./data/env.py`
   <details>
   <summary>env.py</summary>

   ```
   class Env:
    endpoint = '{your appwrite endpoint}'
    project_id = '{your appwrite project id}'
    api_key = '{your appwrite secret api key}'
    database_id = '{database id where collection is found}'
   ```
   
   </details>
10. If using Firebase, add `admin_key.json` to `./data/{admin_key.json}`
11. Update `__C.YOLO.CLASSES` directory in `./core/config.py` from step 7
12. Run `python detectvideo.py --weights ./checkpoints/{weights} --video ./data/{video} --database {database} --area {area}`
13. Press Q to exit

## Features
- TensorFlow
- Yolov4
- Realtime Database (Appwrite or Firebase)

## Screenshot
![](/screenshot.png)
