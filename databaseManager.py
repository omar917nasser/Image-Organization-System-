from pymongo import MongoClient
from datetime import datetime
import torch
import numpy as np
from bson import ObjectId
import json
from aiModels import DEVICE
from config import CONNECTION_URI, DATABASE_NAME

class MongoDBManager:
    def __init__(self, connection_uri=CONNECTION_URI, database_name=DATABASE_NAME):
        """Initialize MongoDB connection with connection pooling."""
        try:
            self.client = MongoClient(connection_uri, maxPoolSize=50, minPoolSize=10)
            self.db = self.client[database_name]
            self.faces_collection = self.db.imageData
            
            # Test connection
            self.client.admin.command('ping')
            print("Successfully connected to MongoDB!")
            
        except Exception as e:
            print(f"Error connecting to MongoDB: {e}")
            raise
    
# In databaseManager.py

    def save_new_person(self, embedding, name_label=None, image_path=None):
        """Save a new person to the database."""
        person_id = self.generatePersonID()

        try:
            embedding_list = self._tensor_to_list(embedding)
            person_doc = {
                "person_id": person_id,
                "name_label": name_label,
                "embeddings": [embedding_list],
                "representative_embedding": embedding_list,  
                "representative_image_paths": [image_path] if image_path else [] # This line is critical
            }
            result = self.faces_collection.insert_one(person_doc)
            print(f"Successfully saved new person with ID: {person_id}")
            return person_id
        except Exception as e:
            print(f"Error saving new person: {e}")
            raise

    def add_embedding_to_person(self, person_id, embedding, image_path):
        """Add a new embedding and image path, then rebuild representative embedding."""
        try:
            embedding_list = self._tensor_to_list(embedding)
            # Push new data
            self.faces_collection.update_one(
                {"person_id": person_id},
                {
                    "$push": {
                        "embeddings": embedding_list,
                        "representative_image_paths": image_path
                    }
                }
            )
            # Recompute and update representative embedding
            self._recompute_representative_embedding(person_id)
            print(f"Successfully added embedding and updated representative for {person_id}")
            return True
        except Exception as e:
            print(f"Error adding embedding to person: {e}")
            raise

    def getPerson(self, person_id):
        """Retrieve a person by their ID."""
        try:
            person = self.faces_collection.find_one({"person_id": person_id})
            if not person:
                raise ValueError(f"Person {person_id} not found")
            
            return {
                "name_label": person.get("name_label"),
                "embeddings": [self._list_to_tensor(emb) for emb in person.get("embeddings", [])],
                "representative_embedding": self._list_to_tensor(person.get("representative_embedding", [])),
                "representative_image_paths": person.get("representative_image_paths", [])
            }
        
        except Exception as e:
            print(f"Error retrieving person: {e}")
            raise
    
    def update_person_name(self, person_id, new_name):
        """Update the name label of a person."""
        try:
            result = self.faces_collection.update_one(
                {"person_id": person_id},
                {"$set": {"name_label": new_name}}
            )
            if result.modified_count > 0:
                print(f"Successfully updated name for person {person_id}")
                return True
            else:
                print(f"No person found with ID {person_id}")
                return False
        except Exception as e:
            print(f"Error updating person name: {e}")
            raise

    def merge_persons(self, target_id, source_ids):
        """Merge multiple persons into one, then rebuild representative embedding."""
        try:
            target = self.faces_collection.find_one({"person_id": target_id})
            if not target:
                raise ValueError(f"Target person {target_id} not found")
            sources = list(self.faces_collection.find({"person_id": {"$in": source_ids}}))
            if len(sources) != len(source_ids):
                raise ValueError("Some source persons not found")
            # Combine lists
            all_embeddings = target.get("embeddings", [])
            all_paths = target.get("representative_image_paths", [])
            for source in sources:
                all_embeddings.extend(source.get("embeddings", []))
                all_paths.extend(source.get("representative_image_paths", []))
            # Update target
            self.faces_collection.update_one(
                {"person_id": target_id},
                {"$set": {"embeddings": all_embeddings, "representative_image_paths": all_paths}}
            )
            # Remove sources
            self.faces_collection.delete_many({"person_id": {"$in": source_ids}})
            # Recompute new representative embedding
            self._recompute_representative_embedding(target_id)
            print(f"Successfully merged {len(source_ids)} into {target_id} in Database.")
            return True
        except Exception as e:
            print(f"Error merging persons: {e}")
            raise

    def get_all_persons(self):
        """Retrieve all persons from the database."""
        try:
            persons = {}
            for doc in self.faces_collection.find({}):
                person_id = doc["person_id"]
                persons[person_id] = {
                    "name_label": doc.get("name_label"),
                    "embeddings": [self._list_to_tensor(emb) for emb in doc.get("embeddings", [])],
                    "representative_embedding": self._list_to_tensor(doc.get("representative_embedding", [])),
                    "representative_image_paths": doc.get("representative_image_paths", [])
                }
            return persons
        except Exception as e:
            print(f"Error retrieving persons: {e}")
            raise


    def _recompute_representative_embedding(self, person_id):
        """Recompute representative embedding as the normalized average of all embeddings."""
        doc = self.faces_collection.find_one({"person_id": person_id}, {"embeddings": 1})
        if not doc or "embeddings" not in doc:
            raise ValueError(f"Person {person_id} not found or has no embeddings")

        # Convert lists to tensors and stack
        tensors = [self._list_to_tensor(e) for e in doc["embeddings"]]
        stacked = torch.stack(tensors, dim=0).float()
        # Compute centroid
        centroid = torch.mean(stacked, dim=0)
        # Normalize
        rep_embed = torch.nn.functional.normalize(centroid, p=2, dim=0)
        rep_list = self._tensor_to_list(rep_embed)
        # Update
        self.faces_collection.update_one(
            {"person_id": person_id},
            {"$set": {"representative_embedding": rep_list}}
        )

    def generatePersonID(self):
        """Generate a unique person ID using timestamp and random string."""
        try:
            # Get current timestamp
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            # Generate a random string using ObjectId
            random_str = str(ObjectId())[-6:]
            # Combine timestamp and random string
            person_id = f"P{timestamp}{random_str}"
            
            # Verify the ID is unique
            while self.faces_collection.find_one({"person_id": person_id}):
                random_str = str(ObjectId())[-6:]
                person_id = f"P{timestamp}{random_str}"
            
            return person_id
            
        except Exception as e:
            print(f"Error generating person ID: {e}")
            raise

    def _tensor_to_list(self, tensor):
        """Convert a PyTorch tensor to a list for MongoDB storage."""
        try:
            if isinstance(tensor, torch.Tensor):
                return tensor.cpu().numpy().tolist()
            elif isinstance(tensor, np.ndarray):
                return tensor.tolist()
            elif isinstance(tensor, list):
                return tensor
            else:
                raise ValueError(f"Unsupported tensor type: {type(tensor)}")
        except Exception as e:
            print(f"Error converting tensor to list: {e}")
            raise

    def _list_to_tensor(self, lst):
        """Convert a list from MongoDB to a PyTorch tensor."""
        try:
            return torch.tensor(lst, device=DEVICE)
        except Exception as e:
            print(f"Error converting list to tensor: {e}")
            raise 

    def close(self):
        """Safely close the MongoDB connection."""
        try:
            if hasattr(self, 'client'):
                self.client.close()
                print("MongoDB connection closed successfully")
        except Exception as e:
            print(f"Error closing MongoDB connection: {e}")