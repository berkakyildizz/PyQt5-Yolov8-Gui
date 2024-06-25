# PyQt5-Yolov8-Onnx-Gui

[![yolov8nonnx](https://img.shields.io/badge/Model%201-Download-brightgreen)](https://drive.google.com/file/d/1eCX7RFXoYOAbkrxWQopf7kRasvAFTZPV/view?usp=drive_link)
[![yolov8monnx](https://img.shields.io/badge/Model%202-Download-brightgreen)](https://drive.google.com/file/d/1aR6F0mMLgyb8wof3fgfEe-wbZIuFEvwW/view?usp=drive_link)
[![yolov8lonnx](https://img.shields.io/badge/Model%203-Download-brightgreen)](https://drive.google.com/file/d/1oIZzHVXNa1h7_oCoaCXisfGQPgNxfqAH/view?usp=drive_link)
[![yolov8xonnx](https://img.shields.io/badge/Model%204-Download-brightgreen)](https://drive.google.com/file/d/1OULaYUwwkUDFBBIgwRJiz5YIAj3tyv8t/view?usp=drive_link)

<h4>
  This repo is specially prepared for users who want to run onnx models. The program still has shortcomings and these can be improved. However, if you want to contribute to development, I wouldn't say no. I would be pleased.

  The aim of this project is to enable CUDA-based operation of onnx models in an interface and human detection. The detections made can be monitored instantly in the interface created with PyQt5. 

  You can select 4 onnx models via the interface, then add and run your rtsp camera or local webcam via the code. 

  Additionally, this interface provides the opportunity to detect objects in live streaming and use onnx models. 

  You can upload your own onnx models here and change the code. 

  Since it is difficult to find such properly working interfaces in the market, I am sharing my own project with you. Of course, the project has shortcomings and these will be improved in the future. If you would like to contribute please feel free to do so.
</h4>

![](https://github.com/berkakyildizz/PyQt5-Yolov8-Gui/blob/main/gif.gif?raw=true)

![](https://github.com/berkakyildizz/PyQt5-Yolov8-Gui/blob/main/icon/aaaaaaa.png?raw=true)


<h4>
  Since it is a program that I am currently distributing, it works and works. I will write down all the changes you need to make in the code. This will save you from a huge burden.
</h4>

## Installation

<h4>
  To run this project, you must first have Python and pip installed on your computer.
  To download this project, you can use the `git clone` command as follows:
</h4>

```sh
https://github.com/berkakyildizz/PyQt5-Yolov8-Gui.git
```
<h4>
  After downloading the project files and installing the required libraries, you can start the GUI by running the following command
</h4>

```sh
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia
# After 
pip install -r requirements.txt
pip install onnx
pip install onnxruntime-gpu
```
<h4>
  If CUDA is not installed on your computer, it will run on CPU models and you can see it as CPUExecutionProvider in the interface. However, if you install CUDA and cudnn, you can run it via CUDAExecutionProvider. My suggestion is to use CUDA.
  I am using CUDA 11.8, cudnn 8.9.2 versions. You can download and install them from the NVIDIA official website.
</h4>

## Run

<h4>
  First, if you have a database and want to record the findings, you will need to make a few changes. If you do not have a database connection, you can run and use the code by canceling the DatabaseManager part.
  

```python
server = 'server_ip'
database = 'database_name'
username = 'username'
password = 'password'
```
   
   You must fill these values ​​with your own database values ​​in the `DatabaseManager` class in the code block.
   After that, 

   `self.camera_box.addItems(['your_rtsp_ip'])`

   You have to add own rtsp adresses in camera_box. For example: 'rtsp://admin:admin1admin1@192.168.1.108:554/cam/realmonitor?channel=1&subtype=1'

   Then you need to give your own url path to camera_url in the start_camera function in the MainWindow class.

   `camera_url = 'enter_rtsp_addresses'`

    The changes you will make to adapt it to yourself are over.
</h4>

<h4>
  After downloading the project files and installing the required libraries, you can start the GUI by running the following command:
</h4>

```sh
python main.py
```
## Deploy

#### To convert the program to an exe file you must:

First, install `pyinstaller` by running the following command:

```sh
pip install pyinstaller
```

After installing pyinstaller, enter the terminal and run:

```sh
pyinstaller main.spec
```

