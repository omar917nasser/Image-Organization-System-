# AI model initialization and configuration for face detection and recognition
import torch
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1

# Set device for model inference (GPU if available, else CPU)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Initialize YOLO model for face detection
YOLO_MODEL = None
try:
    YOLO_MODEL = YOLO('yolov11l-face.pt')  
    YOLO_MODEL.to(DEVICE)
    print("YOLO model 'yolov11l-face.pt' loaded successfully.")
except Exception as e:
    print(f"Error loading YOLO model 'yolov11l-face.pt': {e}. Please ensure the model file is in the project root.")

# Initialize FaceNet model for face embedding generation
FACENET_MODEL = None
try:
    FACENET_MODEL = InceptionResnetV1(pretrained='vggface2', device=DEVICE).eval()
    print("FaceNet model loaded successfully.")
except Exception as e:
    print(f"Error loading FaceNet model: {e}")