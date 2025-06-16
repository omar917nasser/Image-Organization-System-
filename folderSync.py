import os
import shutil

from databaseManager import MongoDBManager
from config import CONNECTION_URI, DATABASE_NAME # Import CONNECTION_URI and DATABASE_NAME

# Initialize DB manager here, as folderSync needs to query person names
# Ensure MongoDBManager is initialized with connection details if it requires them
db_manager = MongoDBManager(connection_uri=CONNECTION_URI, database_name=DATABASE_NAME)

def rename_folder_on_disk(old_name_label: str, new_name_label: str, output_dir: str) -> bool:
    """
    Renames a person's folder in the specified output_dir.
    """
    old_folder_path = os.path.join(output_dir, old_name_label)
    new_folder_path = os.path.join(output_dir, new_name_label)

    if os.path.exists(old_folder_path):
        try:
            os.rename(old_folder_path, new_folder_path)
            print(f"Renamed folder from '{old_name_label}' to '{new_name_label}'")
            return True
        except Exception as e:
            print(f"Error renaming folder '{old_name_label}' to '{new_name_label}': {e}")
            return False
    else:
        print(f"Old folder '{old_name_label}' not found at '{old_folder_path}'.")
        return False

def merge_person_folders(target_person_id: str, source_person_ids: list, output_dir: str) -> bool:
    """
    Merges content from source person folders into the target person's folder
    within the specified output_dir and then deletes the source folders.
    """
    try:
        targetData = db_manager.getPerson(target_person_id)
        if not targetData:
            print(f"Target person with ID '{target_person_id}' not found in DB.")
            return False
        
        targetFolder = target_person_id if not targetData["name_label"] else targetData["name_label"]
        targetFolderPath = os.path.join(output_dir, targetFolder)
        print(f"Target folder path: {targetFolderPath}")

        sourceFolder = []
        for source in source_person_ids:
            try:
                sourceData = db_manager.getPerson(source)
                if not sourceData:
                    print(f"Source person with ID '{source}' not found in DB, Skipping.")
                    continue
            except Exception as e:
                print(f"Error in the retriving source {source} error is : {e}")
            
            sourceIndvFolder = source if not sourceData["name_label"] else sourceData["name_label"]
            sourceFolder.append(sourceIndvFolder)

        image_exts = {'.jpg', '.jpeg', '.png'}

        for Folder in sourceFolder:
            try:
                folderPath = os.path.join(output_dir, Folder)
                if not os.path.exists(folderPath):
                    print(f"Source folder '{folderPath}' does not exist, skipping.")
                    continue

                for filename in os.listdir(folderPath):
                    # build full path to the file
                    filepath = os.path.join(folderPath, filename)
                    
                    if os.path.isfile(filepath):
                        _, ext = os.path.splitext(filename)
                        if ext.lower() in image_exts:
                            try:
                                shutil.copy2(filepath,targetFolderPath)
                                print(f"Photo {filename} is copied to {targetFolderPath}")
                            except Exception as e:
                                print(f" Error Coping Photo {filename} : {e}")
                shutil.rmtree(folderPath)
                print(f"Merged content from '{Folder}' into '{targetFolder}' and removed '{Folder}'.")
            except Exception as e:
                print(f"Error merging folder '{Folder}': {e}")
        return True
    except Exception as e:
        print(f"Error in merge_person_folders: {e}")
        return False