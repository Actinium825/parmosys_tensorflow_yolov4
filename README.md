# Python App for Project Parmosys (Video Object Detection)

A fork of https://github.com/hunglc007/tensorflow-yolov4-tflite for my thesis project for my 4th year at university to
detect available parking spaces and saving the data in a realtime database, either Appwrite or Firebase. This Python App
works in tandem with Parmosys Flutter Mobile App to display the availability. I removed any unused functions, fixed, and
updated the rest so that the latest Python and respective packages could always be used.

## Contents

- [Steps to run](#steps-to-run)
- [Features](#features)
- [Credits](#credits)
- [Screenshot](#screenshot)

## Steps to run

1. Install latest python version
2. Run `python -m venv venv` to create a virtual environment
3. Activate the virtual environment by running `source venv/bin/activate`
4. Double check the virtual environment is using the latest python version with `python --version`
5. Run `pip install -r requirements.txt`
6. Download my weights and video for detection
   <details>
   <summary>Link</summary>

   https://drive.proton.me/urls/ZHDT168A0G#MWuu21jvbTCC

   </details>

7. Place `22class` weights folder in `./checkpoints/22class`
8. Place `test2.wmv` video in `./data/test2.wmv`
9. Skip to Step 11 if running without database
10. Setup database if uploading realtime data:
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

11. Run `python detectvideo.py` or `python detectvideo.py --database {appwrite or firebase}` if running with database
12. To change parking space area, update the `area` flag with other options from [Parmosys Flutter](https://github.com/Actinium825/parmosys_flutter) (snakecase)
    - `python detectvideo.py --database {appwrite or firebase} --area college_of_law`
13. Press Q to exit

## Features

- TensorFlow
- Yolov4
- Realtime Database (Appwrite or Firebase)

## Credits

- Forked from https://github.com/hunglc007/tensorflow-yolov4-tflite

## Screenshot

![](/screenshot.png)
