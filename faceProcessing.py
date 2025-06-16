# Core face processing module for face detection, recognition, and person management
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import uuid
from aiModels import YOLO_MODEL, FACENET_MODEL, DEVICE
from databaseManager import MongoDBManager
from config import CONNECTION_URI, DATABASE_NAME, SIMILARITY_THRESHOLD, CONFIDENCE_THRESHOLD
from folderSync import rename_folder_on_disk, merge_person_folders

# Initialize database connection
db_manager = MongoDBManager(connection_uri=CONNECTION_URI, database_name=DATABASE_NAME)

def standardize_image(img):
    """Standardize image for FaceNet input by converting to RGB and applying transforms."""
    # Convert to RGB if needed
    if isinstance(img, np.ndarray):
        if len(img.shape) == 2:  # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    # Convert to PIL Image if needed
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    
    # Define transforms for FaceNet input
    transform = transforms.Compose([
        transforms.Resize((160, 160)),  # FaceNet expects 160x160
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Apply transforms
    img_tensor = transform(img)
    return img_tensor

def detect_faces_yolo(image_path):
    """Detect faces in an image using YOLO with padding for better face capture."""
    try:
        # Read image with PIL first to ensure RGB format
        img_pil = Image.open(image_path).convert('RGB')
        img = np.array(img_pil)
        
        # Get image dimensions
        height, width = img.shape[:2]
        
        # Run YOLO detection with confidence threshold
        results = YOLO_MODEL(img, device=DEVICE, conf=CONFIDENCE_THRESHOLD, save=False)

        bboxes = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Add padding (20% of box size) for better face capture
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
    """Generate face embedding using FaceNet model."""
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
    """Identify a person based on face embedding similarity or create new person if no match."""
    try:
        # Get all persons from database
        persons = db_manager.get_all_persons()
        
        best_match = None
        best_similarity = -1
        
        # Find best matching person based on cosine similarity
        for person_id, person_data in persons.items():
            embedding2 = person_data["representative_embedding"]
            similarity = torch.nn.functional.cosine_similarity(
                embedding1.unsqueeze(0),
                embedding2.unsqueeze(0)
            ).item()
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = person_id

        # Return existing person if similarity exceeds threshold
        if best_similarity >= similarity_threshold:
            if image_path:
                db_manager.add_embedding_to_person(best_match, embedding1, image_path)
            return best_match
        else:
            # Create new person if no match found
            if image_path:
                id = db_manager.save_new_person(embedding1, name_label=None, image_path=image_path)
            else:
                id = db_manager.save_new_person(embedding1, name_label=None)
            return id
        
    except Exception as e:
        print(f"Error identifying person: {e}")
        return None

def get_person_name(person_id):
    """Retrieve name label for a person from database."""
    try:
        person = db_manager.getPerson(person_id)
        return person["name_label"]
    
    except Exception as e:
        print(f"Error getting person name: {e}")
        return None

def update_person_name(person_id, new_name, output_dir):
    """Update person's name in database and rename their folder."""
    try:
        # Get current person data before update
        current_person_data = db_manager.getPerson(person_id)
        old_name_label = current_person_data.get("name_label")
        
        current_folder_name = old_name_label if old_name_label else person_id

        # Update database first
        db_update_success = db_manager.update_person_name(person_id, new_name)

        if db_update_success:
            # Rename folder on disk
            new_folder_name = new_name
            folder_rename_success = rename_folder_on_disk(current_folder_name, new_folder_name, output_dir)
            
            if folder_rename_success:
                print(f"Successfully updated name for person {person_id} and renamed folder.")
                return True
            else:
                print(f"Successfully updated name for person {person_id} in DB, but failed to rename folder.")
                return False
        else:
            print(f"Failed to update name for person {person_id} in database.")
            return False
    except Exception as e:
        print(f"Error updating person name and folder: {e}")
        return False

def merge_persons(target_id, source_ids, output_dir):
    """Merge multiple persons into one target person and combine their folders."""
    try:
        # Verify target person exists
        target_person_data = db_manager.getPerson(target_id)
        if not target_person_data:
            print(f"Target person with ID '{target_id}' not found in DB.")
            return False

        # Verify all source persons exist
        for source in source_ids:
            source_person_data = db_manager.getPerson(source)
            if not source_person_data:
                print(f"Source person with ID '{source}' not found in DB. Skipping.")
                continue

        # Merge folders on disk first
        db_merge_success = merge_person_folders(target_id, source_ids, output_dir) 
        print("Successfully merged persons in folders")

        if db_merge_success:
            # Then merge in database
            folder_merge_success = db_manager.merge_persons(target_id, source_ids)
            
            if folder_merge_success:
                print(f"Successfully merged persons and their folders.")
                return True
            else:
                print(f"Successfully merged persons in folders, but failed to merge DB.")
                return False
        else:
            print(f"Failed to merge persons in database.")
            return False
    except Exception as e:
        print(f"Error merging persons and folders: {e}")
        return False

def close_database():
    """Close database connection."""
    try:
        db_manager.close()
    except Exception as e:
        print(f"Error closing database: {e}") 