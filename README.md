# Azerbaijani Cars License Plates OCR

A deep learning pipeline for detecting and recognizing Azerbaijani vehicle license plates from images and videos using YOLOv11 and CRNN models.

## Features

- **License Plate Detection**: YOLOv11-based plate localization
- **OCR Recognition**: CRNN model for text extraction
- **Image Processing**: Single image analysis with visualization
- **Video Processing**: Real-time plate tracking and recognition in videos

## Installation

```bash
pip install torch torchvision ultralytics huggingface-hub opencv-python pillow numpy matplotlib
```

## Model & Dataset

- **Model**: [Azerbaijani Cars License Plates OCR Model](https://www.kaggle.com/models/vrashad/azerbaijani-cars-license-plates-ocr-model)
- **Dataset**: [Azerbaijani Cars License Plates OCR Dataset](https://www.kaggle.com/datasets/vrashad/azerbaijani-cars-license-plates-ocr-dataset)

Download `crnn_final_model.pth` from the Kaggle model page and place it in the project root directory.

## Usage

### Image Recognition

```python
from image_plate_recognizer import LicensePlateRecognizer

recognizer = LicensePlateRecognizer(
    crnn_model_path='crnn_final_model.pth',
    yolo_confidence=0.5,
    device='cuda'
)

results = recognizer.process_image('your_image.jpg', visualize=True)

for result in results:
    print(f"Plate: {result['text']}")
    print(f"Confidence: {result['detection_confidence']:.2%}")
```

### Video Recognition

```python
from video_plate_recognizer import VideoLicensePlateRecognizer

recognizer = VideoLicensePlateRecognizer(
    crnn_model_path='crnn_final_model.pth',
    device='cuda',
    min_text_length=5,
    detection_conf=0.5
)

recognizer.process_video('input_video.mp4', 'output_video.mp4')
```

## Example Output

```
============================================================
FINAL RESULTS:
============================================================
Plate 1: 77-FP-263
  Detection confidence: 78.50%
  Bbox: (165, 299, 262, 357)
============================================================
```

## Project Structure

```
azerbaijan_plates_ocr/
├── image_plate_recognizer.py    # Image-based plate recognition
├── video_plate_recognizer.py    # Video-based plate recognition
├── model_train_code.ipynb       # Model training notebook
├── crnn_final_model.pth         # Trained CRNN model (download separately)
└── README.md
```

