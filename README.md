<H1 align="center">Automatic Number Plate Detection and Recognition using YOLOv8</H1>


## Steps to run Code

- Clone the repository
```
git clone https://github.com/MuhammadMoinFaisal/Automatic_Number_Plate_Detection_Recognition_YOLOv8.git
```
- Goto the cloned folder.
```
cd Automatic_Number_Plate_Detection_Recognition_YOLOv8
```
- Install the dependecies
```
pip install -e '.[dev]'

```

- Setting the Directory.
```
cd ultralytics/yolo/v8/detect
```


- Downloading a Weights from the Google Drive
```
gdown "https://drive.google.com/uc?id=1dIyJooVaowaNUj0R1Q-HUnu-utiGsEj8&confirm=t"
```
- Downloading a Sample Video from the Google Drive
```
gdown "https://drive.google.com/uc?id=1P-oVR0J35Dw40lzw47sE19oADSW-tyb1&confirm=t"

```
- Run the code with mentioned command below (For Licence Plate Detection).
```
python predict.py model='C:\Users\Afnan\OneDrive\Desktop\Class\FYP1\code\Automatic_Number_Plate_Detection_Recognition_YOLOv8\ultralytics\yolo\v8\detect\train_model\weights\best.pt' source='C:\Users\Afnan\OneDrive\Desktop\Class\FYP1\code\Automatic_Number_Plate_Detection_Recognition_YOLOv8\runs\test_subject\demo.mp4'
```

- For Licence Plate Detection and Recognition

- Download the Updated predict.py file from the drive
```
gdown "https://drive.google.com/uc?id=1S6GkQcDq8W0ThaUeZ708UegHIRiVWzTo&confirm=t"
```
- Run the code with mentioned command below (For Licence Plate Detection and Recognition).
```
python predict.py model="C:\Users\Afnan\OneDrive\Desktop\Class\FYP1\code\Automatic_Number_Plate_Detection_Recognition_YOLOv8\ultralytics\yolo\v8\detect\train_model\weights\best.pt" source=0
```
```
this to be executed in pi
python predict.py model="/home/afnan02/Desktop/Automatic_Number_Plate_Detection_Recognition_YOLOv8/ultralytics/yolo/v8/detect/train_model/weights/best.pt" source=0
```

#C:\Users\Afnan\OneDrive\Desktop\Class\FYP1\code\Automatic_Number_Plate_Detection_Recognition_YOLOv8\runs\test_subject\malaysian_vehicle_dataset\test\images"
### RESULTS

#### Licence Plate  Detection and Recognition  
![](./ultralytics/figure1.png)

#### Licence Plate  Detection and Recognition

![](./ultralytics/figure3.png)

### Watch the Complete Step by Step Explanation

- Video Tutorial Link  [`YouTube Link`](https://www.youtube.com/watch?v=lkAJBSmIL50&t=5s)


[![Watch the Complete Tutorial for the Step by Step Explanation](https://img.youtube.com/vi/lkAJBSmIL50/0.jpg)]([https://www.youtube.com/watch?v=StTqXEQ2l-Y](https://www.youtube.com/watch?v=lkAJBSmIL50)
