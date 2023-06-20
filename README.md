## Final Project Startup Campus

#### For detailed explanation can see pdf file on documentationproject folder

Team 1 - Cuan Hunter </br>
Topic : Real-time Object Detection Indonesian Sign Language With YOLOv5-MobileNetv3s

#### View on Streamlit Cloud
*Problem when deploy the program is webcam can't start because of the use of the server is free so there are limitations for accessing the webcam via the cloud
![image](https://github.com/hardiantots/FP_YOLOv5-MBv3S/assets/111510893/57ca74c3-2c24-428c-b870-b4ca1b7734d5)

### Graph result of Training Model (Loss & Evaluation Metric)
![image](https://github.com/hardiantots/FP_YOLOv5-MBv3S/assets/111510893/2031afec-2520-4fff-a4d0-f95d513a5041)
More epoch = Decrease loss
More epoch = Increase the Metrics

### Comparison Evaluation Metrics between YOLOv5 Default & YOLOv5 MBv3s
![image](https://github.com/hardiantots/FP_YOLOv5-MBv3S/assets/111510893/d89399c4-647e-417f-a1c2-9b6fa9e501db)

### Conclusion from this Final Project
1. The level of responsiveness in detecting objects related to sign language is still lacking. The main influencing factor is the dataset we have, because our group has problems in the diversity and amount of data in each class.
2. The dataset that will be carried out by the modeling process plays an important role for the success of this project
3. By changing this yolov5 backbone to MobileNetv3small, the modeling results will be lighter and can be applied to systems that have limited memory
