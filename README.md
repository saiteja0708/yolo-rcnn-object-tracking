# Object Detection: YOLOv5 vs R-CNN (ResNet-18)

Comparison of YOLOv5 and R-CNN (ResNet-18 backbone) for object detection on the **Pascal VOC 2012 dataset** (20 classes).  
Evaluated on training loss, validation loss, and mean Average Precision (mAP @ 0.5).

---

## 📊 Results

| Metric | YOLOv5 | R-CNN (ResNet-18) |
|---|---|---|
| Training Loss | 0.014 | 0.777 |
| Validation Loss | 0.030 | 0.803 |
| **mAP @ 0.5** | **0.605** | 0.543 |

YOLOv5 outperformed R-CNN in both accuracy and training efficiency, achieving 11.4% higher mAP with significantly lower loss.

---

## 🛠️ Tools & Libraries

- Python, PyTorch
- YOLOv5 (Ultralytics)
- Torchvision (R-CNN with ResNet-18 backbone)
- Pascal VOC 2012 dataset (20 object classes)

---

## 📂 File Structure

| File | Description |
|---|---|
| `train.py` | Training script for R-CNN model |
| `training_and_validation.py` | Combined training and validation loop |
| `testing.py` | Inference and mAP evaluation |
| `results.xlsx` | mAP comparison results |
| `losses.xlsx` | Training and validation loss data |
| `WiSe23_Group20_Report_2.pdf` | Full project report |

---

## 👤 My Contribution

I implemented the R-CNN training pipeline using PyTorch and Torchvision, handled the model benchmarking and comparative analysis against YOLOv5, and produced the quantitative results documented in this repository.

*(Edit this section to reflect what you personally built)*

---

## ▶️ How to Run

```bash
pip install torch torchvision
python train.py
python training_and_validation.py
python testing.py
```
