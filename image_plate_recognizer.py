import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import cv2
import numpy as np
from pathlib import Path


class ResizeWithPad:
    def __init__(self, target_height=32, target_width=128, fill_value=0):
        self.target_height = target_height
        self.target_width = target_width
        self.fill_value = fill_value

    def __call__(self, img):
        w, h = img.size

        aspect = w / h
        target_aspect = self.target_width / self.target_height

        if aspect > target_aspect:
            new_width = self.target_width
            new_height = int(self.target_width / aspect)
        else:
            new_height = self.target_height
            new_width = int(self.target_height * aspect)

        img = img.resize((new_width, new_height), Image.LANCZOS)

        new_img = Image.new('L', (self.target_width, self.target_height), self.fill_value)

        paste_x = (self.target_width - new_width) // 2
        paste_y = (self.target_height - new_height) // 2
        new_img.paste(img, (paste_x, paste_y))

        return new_img


class CRNN(nn.Module):
    def __init__(self, num_classes, dropout=0.3):
        super(CRNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Dropout2d(0.2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Dropout2d(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Dropout2d(0.3),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1))
        )

        self.rnn = nn.LSTM(512 * 2, 256, bidirectional=True, num_layers=2, batch_first=True, dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)

        batch, channels, height, width = x.size()
        x = x.reshape(batch, channels * height, width)
        x = x.permute(0, 2, 1)

        x, _ = self.rnn(x)
        x = self.dropout(x)
        x = self.classifier(x)

        x = x.permute(1, 0, 2)
        x = nn.functional.log_softmax(x, dim=2)

        return x


class LicensePlateRecognizer:
    def __init__(self, crnn_model_path, yolo_confidence=0.5, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        print("Loading YOLO model...")
        yolo_model_path = hf_hub_download(
            repo_id="morsetechlab/yolov11-license-plate-detection",
            filename="license-plate-finetune-v1x.pt"
        )
        self.yolo_model = YOLO(yolo_model_path)
        self.yolo_confidence = yolo_confidence
        print("YOLO model loaded")

        print("Loading CRNN model...")
        checkpoint = torch.load(crnn_model_path, map_location=self.device)
        self.char_to_idx = checkpoint['char_to_idx']
        self.idx_to_char = checkpoint['idx_to_char']
        self.num_classes = checkpoint['num_classes']
        self.blank_label = checkpoint['blank_label']
        self.img_height = checkpoint.get('img_height', 32)
        self.img_width = checkpoint.get('img_width', 128)

        self.crnn_model = CRNN(num_classes=self.num_classes, dropout=0.3).to(self.device)
        self.crnn_model.load_state_dict(checkpoint['model_state_dict'])
        self.crnn_model.eval()
        print("CRNN model loaded")

        self.transform = transforms.Compose([
            ResizeWithPad(self.img_height, self.img_width),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def detect_plates(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")

        results = self.yolo_model.predict(source=image_path, conf=self.yolo_confidence, verbose=False)

        plates = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])

                if x1 >= x2 or y1 >= y2:
                    continue

                plate_crop = image[y1:y2, x1:x2]
                plates.append({
                    'image': plate_crop,
                    'bbox': (x1, y1, x2, y2),
                    'confidence': confidence
                })

        return plates

    def decode_prediction(self, output):
        output = output.permute(1, 0, 2)
        _, max_index = torch.max(output, dim=2)

        raw = max_index[0].tolist()
        decoded_seq = []
        prev_char = None
        for char in raw:
            if char != self.blank_label and char != prev_char:
                decoded_seq.append(char)
            prev_char = char

        pred_text = ''.join([self.idx_to_char[idx] for idx in decoded_seq])
        return pred_text

    def recognize_plate(self, plate_image):
        plate_pil = Image.fromarray(cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB)).convert('L')

        plate_tensor = self.transform(plate_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.crnn_model(plate_tensor)
            plate_text = self.decode_prediction(output)

        return plate_text

    def process_image(self, image_path, visualize=False):
        print(f"\nProcessing: {image_path}")

        plates = self.detect_plates(image_path)

        if not plates:
            print("No plates detected")
            return []

        print(f"Detected {len(plates)} plate(s)")

        results = []
        for i, plate_data in enumerate(plates, 1):
            print(f"\nPlate {i}:")
            print(f"  Bbox: {plate_data['bbox']}")
            print(f"  Detection confidence: {plate_data['confidence']:.2%}")

            plate_text = self.recognize_plate(plate_data['image'])
            print(f"  Recognized text: {plate_text}")

            results.append({
                'text': plate_text,
                'bbox': plate_data['bbox'],
                'detection_confidence': plate_data['confidence'],
                'plate_image': plate_data['image']
            })

        if visualize:
            self.visualize_results(image_path, results)

        return results

    def visualize_results(self, image_path, results):
        import matplotlib.pyplot as plt

        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, len(results) + 1, figsize=(5 * (len(results) + 1), 5))
        if len(results) == 0:
            axes = [axes]

        axes[0].imshow(image_rgb)
        axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        for result in results:
            x1, y1, x2, y2 = result['bbox']
            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(image_rgb, result['text'], (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        axes[0].imshow(image_rgb)

        for i, result in enumerate(results, 1):
            plate_rgb = cv2.cvtColor(result['plate_image'], cv2.COLOR_BGR2RGB)
            axes[i].imshow(plate_rgb)
            axes[i].set_title(f"Plate: {result['text']}\nConf: {result['detection_confidence']:.2%}",
                              fontsize=12, fontweight='bold')
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()


def main():
    CRNN_MODEL_PATH = 'crnn_final_model.pth'
    IMAGE_PATH = 'sample_image_1.jpg'

    recognizer = LicensePlateRecognizer(
        crnn_model_path=CRNN_MODEL_PATH,
        yolo_confidence=0.5,
        device='cuda'
    )

    results = recognizer.process_image(IMAGE_PATH, visualize=True)

    print(f"\n{'=' * 60}")
    print("FINAL RESULTS:")
    print(f"{'=' * 60}")
    for i, result in enumerate(results, 1):
        print(f"Plate {i}: {result['text']}")
        print(f"  Detection confidence: {result['detection_confidence']:.2%}")
        print(f"  Bbox: {result['bbox']}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()