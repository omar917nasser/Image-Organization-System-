import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import uuid
from aiModels import YOLO_MODEL, FACENET_MODEL, DEVICE
from databaseManager import MongoDBManager
from config import CONNECTION_URI, DATABASE_NAME, SIMILARITY_THRESHOLD, CONFIDENCE_THRESHOLD

# Initialize database connection
db_manager = MongoDBManager(
    connection_uri=CONNECTION_URI,
    database_name=DATABASE_NAME
)

# Define image standardization transform
def standardize_image(img):
    """Standardize image for FaceNet input."""
    # Convert to RGB if needed
    if isinstance(img, np.ndarray):
        if len(img.shape) == 2:  # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    # Convert to PIL Image if needed
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((160, 160)),  # FaceNet expects 160x160
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Apply transforms
    img_tensor = transform(img)
    return img_tensor


def detect_faces_yolo(image_path):
    """Detect faces in an image using YOLO."""
    try:
        # Read image with PIL first to ensure RGB format
        img_pil = Image.open(image_path).convert('RGB')
        img = np.array(img_pil)
        
        # Get image dimensions
        height, width = img.shape[:2]
        
        results = YOLO_MODEL(img, device=DEVICE, conf=CONFIDENCE_THRESHOLD)

        bboxes = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Add padding (20% of box size)
                box_width = x2 - x1
                box_height = y2 - y1
                padding_x = int(box_width * 0.2)
                padding_y = int(box_height * 0.2)
                
                # Apply padding while ensuring we don't go out of image bounds
                x1 = max(0, x1 - padding_x)
                y1 = max(0, y1 - padding_y)
                x2 = min(width, x2 + padding_x)
                y2 = min(height, y2 + padding_y)
                
                bboxes.append([x1, y1, x2, y2])
        return bboxes
    except Exception as e:
        print(f"Error during YOLO face detection on {image_path}: {e}")
        return []
    
def get_face_embedding(face_image):
    """Get face embedding using FaceNet."""
    try:
        # Standardize the image
        face_tensor = standardize_image(face_image)
        face_tensor = face_tensor.unsqueeze(0).to(DEVICE)  # Add batch dimension

        with torch.no_grad():
            embedding = FACENET_MODEL(face_tensor)
        return embedding.squeeze(0)
    except Exception as e:
        print(f"Error getting face embedding: {e}")
        return None
    



def identify_person(embedding1, similarity_threshold=SIMILARITY_THRESHOLD, image_path=None):
    """Identify a person based on their face embedding."""
    try:
        # Get all persons from database
        persons = db_manager.get_all_persons()
        
        best_match = None
        best_similarity = -1
        
        for person_id, person_data in persons.items():
            embedding2 = person_data["representative_embedding"]
            similarity = torch.nn.functional.cosine_similarity(
                embedding1.unsqueeze(0),
                embedding2.unsqueeze(0)
            ).item()
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = person_id

        if best_similarity >= similarity_threshold:
            # Add new embedding to existing person and update the representative embedding
            if image_path:
                db_manager.add_embedding_to_person(best_match, embedding1, image_path)
            return best_match
        else:
            if image_path:
                id =db_manager.save_new_person(embedding1, image_path)
            return id
        
    except Exception as e:
        print(f"Error identifying person: {e}")
        return None
    

def get_person_name(person_id):
    """Get the name label for a person."""
    try:
        person = db_manager.getPerson(person_id)
        return person["name_label"]
    
    except Exception as e:
        print(f"Error getting person name: {e}")
        return None

def update_person_name(person_id, new_name):
    """Update the name label for a person."""
    try:
        return db_manager.update_person_name(person_id, new_name)
    except Exception as e:
        print(f"Error updating person name: {e}")
        return False

def merge_persons(target_id, source_ids):
    """Merge multiple persons into one."""
    try:
        return db_manager.merge_persons(target_id, source_ids)
    except Exception as e:
        print(f"Error merging persons: {e}")
        return False

def close_database():
    """Close the database connection."""
    try:
        db_manager.close()
    except Exception as e:
        print(f"Error closing database: {e}") 