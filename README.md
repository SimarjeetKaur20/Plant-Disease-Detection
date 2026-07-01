# 🌿 Plant Disease Detection using Knowledge Distillation

A deep learning project that detects plant leaf diseases using Knowledge Distillation. A powerful Teacher model (ResNet18) transfers its knowledge to a lightweight Student model (MobileNetV2), achieving high accuracy while reducing model size and inference time.

---

## 📌 Features

- Detects 15 different plant diseases
- Knowledge Distillation (Teacher → Student)
- Teacher Model: ResNet18
- Student Model: MobileNetV2
- Image Prediction
- Confusion Matrix
- Classification Report
- Training Curves
- Top-1 & Top-5 Accuracy
- Lightweight deployment model

---

## 📂 Dataset

PlantVillage Dataset

Classes Included:

- Pepper Bell Bacterial Spot
- Pepper Bell Healthy
- Potato Early Blight
- Potato Healthy
- Potato Late Blight
- Tomato Target Spot
- Tomato Mosaic Virus
- Tomato Yellow Leaf Curl Virus
- Tomato Bacterial Spot
- Tomato Early Blight
- Tomato Healthy
- Tomato Late Blight
- Tomato Leaf Mold
- Tomato Septoria Leaf Spot
- Tomato Spider Mites

---

## 🛠 Technologies Used

- Python
- PyTorch
- Torchvision
- OpenCV
- NumPy
- Matplotlib
- Scikit-Learn
- Pillow

---

## 📁 Project Structure

```
PlantDiseaseDetection/
│
├── backend/
│   ├── config.py
│   ├── data_loader.py
│   ├── teacher_model.py
│   ├── student_model.py
│   ├── train_teacher.py
│   ├── train_student.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
│
├── dataset/
├── models/
├── results/
├── notebooks/
├── research_papers/
└── README.md
```

---

## 🚀 Training

### Train Teacher

```bash
python backend/train_teacher.py
```

### Train Student

```bash
python backend/train_student.py
```

### Evaluate Models

```bash
python backend/evaluate.py
```

### Predict Single Image

```bash
python backend/predict.py --image "path_to_image.jpg" --model student
```

---

## 📊 Results

### Teacher Model

- Model: ResNet18
- Top-1 Accuracy: 73.01%
- Top-5 Accuracy: 96.64%
- Model Size: 43.7 MB

### Student Model

- Model: MobileNetV2
- Top-1 Accuracy: 95.32%
- Top-5 Accuracy: 99.87%
- Model Size: 9.7 MB

---

## 📈 Model Comparison

| Metric | Teacher | Student |
|---------|----------|----------|
| Top-1 Accuracy | 73.01% | **95.32%** |
| Top-5 Accuracy | 96.64% | **99.87%** |
| Model Size | 43.7 MB | **9.7 MB** |
| Parameters | 11.4 Million | **2.56 Million** |
| Compression | 4.5× | ✅ |
| Speedup | 1.2× | ✅ |

---

## 📷 Example Prediction

```
Plant       : Potato
Prediction  : Potato Early Blight
Confidence  : 38.02%
```

---

## 📌 Future Improvements

- Streamlit Web Application
- Mobile Deployment
- Real-time Camera Detection
- Additional Crop Diseases
- Explainable AI (Grad-CAM)

---

## 👩‍💻 Author

**Simarjeet Kaur**

B.Tech Computer Science Engineering

GitHub: https://github.com/SimarjeetKaur20

---
