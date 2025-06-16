from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Initialize YOLO model for face detection
YOLO_MODEL = None
try:
    YOLO_MODEL = YOLO('yolov11l-face.pt')  # Switched to the larger model
    YOLO_MODEL.to(DEVICE)
    print("YOLO model 'yolov11l-face.pt' loaded successfully.")
except Exception as e:
    print(f"Error loading YOLO model 'yolov11l-face.pt': {e}. Please ensure the model file is in the project root.")

# Initialize FaceNet model for face recognition
FACENET_MODEL = None
try:
    FACENET_MODEL = InceptionResnetV1(pretrained='vggface2', device=DEVICE).eval()
    print("FaceNet model loaded successfully.")
except Exception as e:
    print(f"Error loading FaceNet model: {e}")