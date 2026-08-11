import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import time

# --- Config ---
MODEL_PATH = "best_drone_bird_classifier.pth"
VIDEO_PATH = "abc.mp4"  # Change extension if needed (e.g., .avi, .mov)
INPUT_SIZE = 224
CLASS_NAMES = ['Bird', 'Drone']
CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence to show prediction

# --- Load Model ---
def load_model(path):
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(1280, 256),
        nn.ReLU6(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(256, 2),
    )
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model

model = load_model(MODEL_PATH)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"Model loaded on {device}")

# --- Transform (same as validation) ---
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# --- Open Video File ---
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"Error: Cannot open video file '{VIDEO_PATH}'")
    exit()

# Get video properties
fps_original = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video loaded: {VIDEO_PATH}")
print(f"FPS: {fps_original}, Total frames: {total_frames}")
print("Press 'q' to quit, 'p' to pause/resume.")

paused = False

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame")
            break

        start_time = time.time()

        # Preprocess the frame
        input_tensor = transform(frame).unsqueeze(0).to(device)  # Add batch dim

        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = probs.max(1)
            confidence = confidence.item() * 100
            predicted_class = CLASS_NAMES[predicted.item()]

        # Calculate FPS
        fps = 1.0 / (time.time() - start_time)

        # --- Draw Results on Frame ---
        # Color: Green for Bird, Red for Drone
        color = (0, 255, 0) if predicted.item() == 0 else (0, 0, 255)

        if confidence >= CONFIDENCE_THRESHOLD * 100:
            label = f"{predicted_class}: {confidence:.1f}%"
        else:
            label = f"Uncertain ({confidence:.1f}%)"
            color = (0, 165, 255)  # Orange for uncertain

        # Black background box for text
        cv2.rectangle(frame, (10, 10), (350, 90), (0, 0, 0), -1)

        # Prediction text
        cv2.putText(frame, label, (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        # FPS text (processing speed)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show frame
        cv2.imshow("Drone vs Bird Classifier - Video", frame)

    # Key controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        paused = not paused
        print("Paused" if paused else "Resumed")

cap.release()
cv2.destroyAllWindows()
print("Video processing stopped.")