# Object Detection using YOLOv5 and R-CNN (ResNet-18)

## 📌 Overview
This project compares **YOLOv5** and **R-CNN (ResNet-18)** for object detection and tracking using the **Pascal VOC 2012 dataset**.  
We evaluate both models on **training loss, validation loss, and mean Average Precision (mAP)**.  
Results show that **YOLOv5 achieved higher accuracy and generalization** compared to R-CNN.

---

## 📂 Dataset
We used the **Pascal VOC 2012 dataset**, which contains 20 classes:  




| Metric          | YOLOv5    | R-CNN (ResNet-18) |
| --------------- | --------- | ----------------- |
| Training Loss   | 0.014     | 0.777             |
| Validation Loss | 0.030     | 0.803             |
| mAP (0.5)       | **0.605** | 0.543             |


YOLOv5 outperformed RCNN with higher mAP and lower losses, indicating better accuracy and generalization.
