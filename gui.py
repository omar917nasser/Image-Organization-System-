import streamlit as st
import os
from PIL import Image
from databaseManager import MongoDBManager
from faceProcessing import update_person_name, merge_persons
from fileOrganizer import process_images

# --- App Configuration ---
st.set_page_config(
    page_title="Face Recognition & Image Organizer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Database Connection ---
@st.cache_resource
def get_db_manager():
    """Caches the database manager for performance."""
    try:
        return MongoDBManager()
    except Exception as e:
        st.error(f"Failed to connect to MongoDB: {e}")
        return None

db_manager = get_db_manager()

# --- Helper Functions ---
def refresh_data():
    """Clears caches to reload data."""
    st.cache_data.clear()

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Dashboard", "Image Processing", "Person Management"],
    key="navigation"
)

# --- Main App ---
if db_manager:
    if page == "Dashboard":
        st.title("📊 Dashboard")
        st.markdown("---")
        
        # --- Key Metrics ---
        st.header("Key Metrics")
        try:
            all_persons = db_manager.get_all_persons()
            num_persons = len(all_persons)
            
            total_images = 0
            for person in all_persons.values():
                total_images += len(person.get("representative_image_paths", []))
            
            col1, col2 = st.columns(2)
            col1.metric("Total People", num_persons)
            col2.metric("Total Images Managed", total_images)
        except Exception as e:
            st.error(f"Could not load metrics: {e}")
        
        st.markdown("---")
        
        # --- Persons Overview ---
        st.header("All Persons in Database")
        if st.button("Refresh Data"):
            refresh_data()
        
        try:
            all_persons_data = db_manager.get_all_persons()
            if all_persons_data:
                person_list = []
                for person_id, data in all_persons_data.items():
                    name = data.get("name_label", "N/A")
                    num_images = len(data.get("representative_image_paths", []))
                    person_list.append({"Person ID": person_id, "Name": name, "Image Count": num_images})
                
                st.dataframe(person_list, use_container_width=True)
            else:
                st.info("No persons found in the database.")
        except Exception as e:
            st.error(f"Error loading person data: {e}")

    elif page == "Image Processing":
        st.title("🖼️ Image Processing")
        st.markdown("---")
        
        st.header("Organize Images by Face")
        
        # --- Directory Selection ---
        input_dir = st.text_input("Enter the Input Directory Path")
        output_dir = st.text_input("Enter the Output Directory Path")
        
        if st.button("Start Processing", key="start_processing"):
            if not os.path.isdir(input_dir):
                st.error("Input directory not found.")
            elif not os.path.isdir(output_dir):
                st.warning("Output directory not found. It will be created.")
                os.makedirs(output_dir, exist_ok=True)
            else:
                with st.spinner("Processing images... This may take a while."):
                    try:
                        stats = process_images(input_dir, output_dir)
                        st.success("Image processing complete!")
                        st.json(stats)
                    except Exception as e:
                        st.error(f"An error occurred during processing: {e}")

    elif page == "Person Management":
        st.title("👥 Person Management")
        st.markdown("---")
        
        try:
            all_persons_data = db_manager.get_all_persons()
            if all_persons_data:
                person_options = {f"{data.get('name_label', 'N/A')} ({person_id})": person_id for person_id, data in all_persons_data.items()}
                
                # --- Rename Person ---
                st.header("Rename a Person")
                selected_person_to_rename = st.selectbox(
                    "Select a Person to Rename", options=list(person_options.keys())
                )
                new_name = st.text_input("Enter the New Name")
                
                if st.button("Update Name"):
                    person_id_to_rename = person_options[selected_person_to_rename]
                    if update_person_name(person_id_to_rename, new_name):
                        st.success(f"Successfully renamed person to '{new_name}'")
                        refresh_data()
                    else:
                        st.error("Failed to rename person.")
                
                st.markdown("---")
                
                # --- Merge Persons ---
                st.header("Merge Persons")
                target_person_key = st.selectbox(
                    "Select Target Person (to merge into)", options=list(person_options.keys())
                )
                source_persons_keys = st.multiselect(
                    "Select Source Persons (to be merged)", options=list(person_options.keys())
                )
                
                if st.button("Merge"):
                    target_id = person_options[target_person_key]
                    source_ids = [person_options[key] for key in source_persons_keys]
                    
                    if target_id in source_ids:
                        st.error("Target person cannot be in the source list.")
                    elif not source_ids:
                        st.warning("Please select at least one source person.")
                    else:
                        if merge_persons(target_id, source_ids):
                            st.success("Successfully merged persons.")
                            refresh_data()
                        else:
                            st.error("Failed to merge persons.")
            else:
                st.info("No persons available to manage.")
        except Exception as e:
            st.error(f"Failed to load person management tools: {e}")

else:
    st.error("Database connection failed. Please check your MongoDB instance and configuration.")

