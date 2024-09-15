# Object-Detection-For-Blind-People-Using-Yolov8-With-Distance-Estimation-of-Multiple-Objects

Overview
Welcome to the Object Detection and Annotation project! This project utilizes YOLO (You Only Look Once) for real-time object detection and integrates text-to-speech capabilities to provide audio feedback about detected objects. It supports both live video feed from a camera and video files, making it versatile for various use cases.
Features
* **Real-time Object Detection:** Detects objects in video feeds or files using YOLOv8.
* **Distance Estimation:** Estimates and announces the distance of detected objects based on their size.
* **Text-to-Speech Integration:** Provides audio feedback about detected objects using `pyttsx3`.
* **Customizable Detection Colors:** Allows for customization of detection bounding box colors.
Requirements
* Python 3.x
* OpenCV
* Pyttsx3
* Ultralytics YOLO
* NumPy
You can install the necessary libraries using pip:

```
bash
```

Copy code
`pip install opencv-python pyttsx3 ultralytics numpy`
Setup
1. **Clone the Repository**

```
git clone https://github.com/Satviksangamkar/Object-Detection-For-Blind-People-Using-Yolov8-With-Distance-Estimation-of-Multiple-Objects-.git
```



2. **Download YOLOv8 Model**
Download the YOLOv8 model weights and place them in the `weights` directory:
   * YOLOv8 Pretrained Weights
Make sure the file is named `yolov8n.pt` and is located in the `weights` directory.
3. **Prepare the Class List**
Ensure you have a `coco.txt` file containing the class names. This file should be placed in the `utils` directory.
4. **Video File Path**
Update the `video_path` variable in the script to point to your video file if you're using a video file instead of a live camera feed.
Usage
For Video Files
