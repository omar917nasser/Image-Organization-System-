import os
import shutil
import cv2
from faceProcessing import detect_faces_yolo, get_face_embedding, identify_person, get_person_name, update_person_name, merge_persons, close_database
from config import SIMILARITY_THRESHOLD

def checkFolders(input_dir: str, output_dir: str):
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' not found.")
        return
    if not os.path.isdir(output_dir):
        print(f"Output directory '{output_dir}' not found. Creating it.")
        os.makedirs(output_dir, exist_ok=True)

def ensure_dir_exists(directory):
    """Ensure a directory exists, create it if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def copy_file_to_destination(source_path: str, dest_dir: str, filename: str) -> bool:
    """Copy a file to a destination directory with proper path handling."""
    try:
        ensure_dir_exists(dest_dir)  # This correctly creates the person's folder
        dest_path = os.path.join(dest_dir, filename)
        
        # The incorrect os.makedirs call has been removed.
        
        if source_path != dest_path:  # Only copy if source and destination are different
            shutil.copy2(source_path, dest_path)
            print(f"Copied {filename} to {dest_dir}")
            return True
        else:
            print(f"Skipping copy - source and destination are the same: {source_path}")
            return False
            
    except Exception as e:
        print(f"Error copying file {filename}: {e}")
        return False

def Facedimensions(detected_faces_bboxes, original_image, filename, image_path: str, identified_person_ids: set):
    unknown_faces = 0
    for bbox in detected_faces_bboxes:
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(original_image.shape[1], x2), min(original_image.shape[0], y2)

        if x1 >= x2 or y1 >= y2:
            print(f"Warning: Invalid bounding box {bbox} for {filename}. Skipping this face.")
            continue

        face_crop_np = original_image[y1:y2, x1:x2]
        face_crop_rgb_np = cv2.cvtColor(face_crop_np, cv2.COLOR_BGR2RGB)
        embedding = get_face_embedding(face_crop_rgb_np)

        if embedding is not None:
            person_id = identify_person(embedding, SIMILARITY_THRESHOLD, image_path)
            if person_id:
                identified_person_ids.add(person_id)
            else:
                unknown_faces += 1
    return unknown_faces

def process_images(input_dir: str, output_dir: str, similarity_threshold=SIMILARITY_THRESHOLD):
    checkFolders(input_dir, output_dir)

    supported_extensions = ('.jpg', '.jpeg', '.png', '.heic')
    processing_stats = {
        "total_files": 0,
        "processed_files": 0,
        "no_faces": 0,
        "Num_of_people": 0,
        "multiple_people": 0,
        "errors": 0
    }

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(supported_extensions):
            print(f"Skipping non-supported file: {filename}")
            continue

        processing_stats["total_files"] += 1
        image_path = os.path.join(input_dir, filename)
        print(f"Processing {image_path}...")

        try:
            original_image = cv2.imread(image_path)
            if original_image is None:
                print(f"Warning: Could not read image {image_path}. Skipping.")
                processing_stats["errors"] += 1
                continue
            
            detected_faces_bboxes = detect_faces_yolo(image_path)

            if not detected_faces_bboxes:
                print(f"No faces detected in {filename}. Moving to '_no_faces'.")
                no_faces_dir = os.path.join(output_dir, "_no_faces")
                if copy_file_to_destination(image_path, no_faces_dir, filename):
                    processing_stats["no_faces"] += 1
                continue
            
            identified_person_ids = set()
            unknown_faces = Facedimensions(detected_faces_bboxes, original_image, filename, image_path, identified_person_ids)

            # Handle the results based on identified and unknown faces
            if not identified_person_ids and unknown_faces > 0:
                # All faces are unknown - create new person(s)
                for i in range(unknown_faces):
                    person_name = f"Person_{processing_stats['Num_of_people'] + 1}"
                    processing_stats["Num_of_people"] += 1
                    person_dir = os.path.join(output_dir, person_name)
                    if copy_file_to_destination(image_path, person_dir, filename):
                        print(f"Created new person {person_name} and moved {filename} there")
                        processing_stats["processed_files"] += 1

            elif identified_person_ids:
                # Copy to all identified person folders
                for person_id in identified_person_ids:
                    person_name = get_person_name(person_id)
                    person_dir = os.path.join(output_dir, person_name if person_name else person_id)
                    if copy_file_to_destination(image_path, person_dir, filename):
                        print(f"Moved {filename} to {person_dir}")
                        processing_stats["processed_files"] += 1

                # If there are unknown faces, create new person(s) for them
                if unknown_faces > 0:
                    for i in range(unknown_faces):
                        person_name = f"Person_{processing_stats['Num_of_people'] + 1}"
                        processing_stats["Num_of_people"] += 1
                        person_dir = os.path.join(output_dir, person_name)
                        if copy_file_to_destination(image_path, person_dir, filename):
                            print(f"Created new person {person_name} and moved {filename} there")
                            processing_stats["processed_files"] += 1

        except Exception as e:
            print(f"Error processing image {filename}: {e}")
            error_dir = os.path.join(output_dir, "_errors")
            if copy_file_to_destination(image_path, error_dir, filename):
                processing_stats["errors"] += 1
    
    return processing_stats
