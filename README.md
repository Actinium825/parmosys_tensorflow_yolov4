# Python App for Project Parmosys (Video Object Detection)

## Steps to run
1. Install python version management `pyenv`
2. Run `pyenv install 3.9.20`
3. Run `python -m venv venv` to create a virtual environment
4. Choose the created virtual environment with version `3.9.20` as the python interpreter for the project
5. Restart the IDE and make sure `venv` is indicated on the terminal and running `python --version` shows `3.9.20`
6. Place your weights folder in `./checkpoints/{weights}`
7. Place your `.names` file in `./data/classes/{names}`
8. Place your video for detection in `./data/{video}`
9. Update `__C.YOLO.CLASSES` directory in `./core/config.py`
10. Run `python detectvideo.py --weights ./checkpoints/{weights} --video ./data/{video}`
11. Press Q to exit

## Features
- TensorFlow
- Yolov4

## TODO
- [ ] Realtime Database

## Screenshot
![](/screenshot.png)
