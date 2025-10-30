import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import cv2
import numpy as np


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


class VideoLicensePlateRecognizer:
    def __init__(self, crnn_model_path, device='cuda', min_text_length=5, detection_conf=0.5):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        print("Loading license plate detection model...")
        yolo_model_path = hf_hub_download(
            repo_id="morsetechlab/yolov11-license-plate-detection",
            filename="license-plate-finetune-v1x.pt"
        )
        self.plate_detector = YOLO(yolo_model_path)
        self.detection_conf = detection_conf
        self.min_text_length = min_text_length
        print("Plate detection model loaded")

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

        self.recognized_plates = {}
        self.plate_history = {}

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

    def draw_plate_corners(self, img, top_left, bottom_right, color=(0, 255, 0), thickness=3, line_length=30):
        x1, y1 = top_left
        x2, y2 = bottom_right
        cv2.line(img, (x1, y1), (x1, y1 + line_length), color, thickness)
        cv2.line(img, (x1, y1), (x1 + line_length, y1), color, thickness)
        cv2.line(img, (x1, y2), (x1, y2 - line_length), color, thickness)
        cv2.line(img, (x1, y2), (x1 + line_length, y2), color, thickness)
        cv2.line(img, (x2, y1), (x2 - line_length, y1), color, thickness)
        cv2.line(img, (x2, y1), (x2, y1 + line_length), color, thickness)
        cv2.line(img, (x2, y2), (x2, y2 - line_length), color, thickness)
        cv2.line(img, (x2, y2), (x2 - line_length, y2), color, thickness)
        return img

    def process_video(self, video_path, output_path):
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_nmr = -1
        ret = True

        while ret:
            frame_nmr += 1
            ret, frame = cap.read()

            if ret:
                results = self.plate_detector.track(frame, conf=self.detection_conf, persist=True, verbose=False)

                if results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                    confidences = results[0].boxes.conf.cpu().numpy()

                    for i, (box, track_id, conf) in enumerate(zip(boxes, track_ids, confidences)):
                        x1, y1, x2, y2 = map(int, box)

                        if track_id not in self.recognized_plates:
                            license_plate_crop = frame[y1:y2, x1:x2, :]

                            if license_plate_crop.size > 0:
                                license_plate_text = self.recognize_plate(license_plate_crop)

                                if license_plate_text and len(license_plate_text) >= self.min_text_length:
                                    if track_id not in self.plate_history:
                                        self.plate_history[track_id] = []

                                    self.plate_history[track_id].append(license_plate_text)

                                    if len(self.plate_history[track_id]) >= 3:
                                        most_common = max(set(self.plate_history[track_id]),
                                                          key=self.plate_history[track_id].count)

                                        if self.plate_history[track_id].count(most_common) >= 2:
                                            self.recognized_plates[track_id] = {
                                                'text': most_common,
                                                'bbox': [x1, y1, x2, y2]
                                            }
                                            print(f"Frame {frame_nmr}: Plate ID {track_id} - Recognized: {most_common}")

                        self.draw_plate_corners(frame, (x1, y1), (x2, y2),
                                                (0, 255, 0), 3, line_length=30)

                        if track_id in self.recognized_plates:
                            plate_text = self.recognized_plates[track_id]['text']

                            text_x = x1
                            text_y = y1 - 10

                            if text_y < 30:
                                text_y = y2 + 30

                            (text_width, text_height), baseline = cv2.getTextSize(
                                plate_text,
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.0,
                                2)

                            bg_x1 = max(0, text_x - 5)
                            bg_y1 = max(0, text_y - text_height - 5)
                            bg_x2 = min(width, text_x + text_width + 5)
                            bg_y2 = min(height, text_y + baseline + 5)

                            overlay = frame.copy()
                            cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
                            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

                            cv2.putText(frame,
                                        plate_text,
                                        (text_x, text_y),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        1.0,
                                        (0, 255, 0),
                                        2)

                out.write(frame)

                if frame_nmr % 30 == 0:
                    print(f"Processed {frame_nmr} frames, recognized {len(self.recognized_plates)} unique plates")

        out.release()
        cap.release()
        print(f"\nVideo processing complete!")
        print(f"Total plates recognized: {len(self.recognized_plates)}")
        for plate_id, data in self.recognized_plates.items():
            print(f"  Plate ID {plate_id}: {data['text']}")


def main():
    CRNN_MODEL_PATH = 'crnn_final_model.pth'
    VIDEO_PATH = 'sample_video_1.mp4'
    OUTPUT_PATH = 'sample_video_1_output.mp4'

    recognizer = VideoLicensePlateRecognizer(
        crnn_model_path=CRNN_MODEL_PATH,
        device='cuda',
        min_text_length=5,
        detection_conf=0.5
    )

    recognizer.process_video(VIDEO_PATH, OUTPUT_PATH)


if __name__ == "__main__":
    main()