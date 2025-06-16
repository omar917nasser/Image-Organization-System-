import streamlit as st
import os
from PIL import Image
from databaseManager import MongoDBManager
from faceProcessing import update_person_name, merge_persons, close_database
from fileOrganizer import process_images
from config import CONNECTION_URI, DATABASE_NAME
import time

# --- App Configuration ---
st.set_page_config(
    page_title="Face Recognition & Image Organizer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Global State for Output Directory ---
# THIS IS THE CRUCIAL PART: Initialize session_state variables at the top-level
# of the script, immediately after st.set_page_config().
if 'output_directory' not in st.session_state:
    st.session_state.output_directory = "" # Initialize with an empty string

# --- Database Connection ---
@st.cache_resource
def get_db_manager():
    """Caches the database manager for performance."""
    try:
        return MongoDBManager(connection_uri=CONNECTION_URI, database_name=DATABASE_NAME)
    except Exception as e:
        st.error(f"Failed to connect to MongoDB: {e}")
        return None

db_manager = get_db_manager()

# --- Helper Functions ---
def refresh_data():
    """Clears caches to reload data."""
    st.cache_data.clear()

def force_rerun():
    """Forces a rerun of the page to refresh the data."""
    st.rerun()

def display_image_preview(person_id, all_persons_data):
    """Helper function to display a person's image preview."""
    person_data = all_persons_data.get(person_id)
    if person_data:
        image_paths = person_data.get("representative_image_paths", [])
        if image_paths:
            first_image_path = image_paths[0]
            if os.path.exists(first_image_path):
                try:
                    image = Image.open(first_image_path)
                    st.image(image, caption=f"Preview for {person_data.get('name_label', person_id)}", use_container_width =True)
                except Exception as e:
                    st.warning(f"Could not load preview: {e}")
            else:
                st.warning(f"Image not found: {first_image_path}")
        else:
            st.info(f"No representative image for {person_data.get('name_label', person_id)}.")


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
        input_dir = st.text_input("Enter the Input Directory Path", placeholder="e.g., C:/Users/YourName/RawImages")
        
        # This input updates the session state. Its value is pre-filled from session_state.
        current_output_dir_input = st.text_input(
            "Enter the Output Directory Path:",
            value=st.session_state.output_directory, # Pre-fill from session_state
            placeholder="e.g., C:/Users/YourName/OrganizedFaces"
        )
        # Update session_state immediately when this input changes
        st.session_state.output_directory = current_output_dir_input

        if st.button("Start Processing", key="start_processing"):
            if not input_dir:
                st.error("Please enter an input folder path.")
            elif not st.session_state.output_directory: # Check from session_state
                st.error("Please enter an output folder path.")
            elif not os.path.exists(input_dir):
                st.error(f"Input folder '{input_dir}' does not exist.")
            else:
                # Ensure output directory exists before processing
                if not os.path.isdir(st.session_state.output_directory):
                    st.warning("Output directory not found. Creating it.")
                    os.makedirs(st.session_state.output_directory, exist_ok=True)

                with st.spinner("Processing images... This may take a while."):
                    try:
                        # Use the output_directory from session_state
                        stats = process_images(input_dir, st.session_state.output_directory)
                        st.success("Image processing complete!")
                        st.json(stats)
                    except Exception as e:
                        st.error(f"An error occurred during processing: {e}")
                refresh_data() # Refresh data after processing

    elif page == "Person Management":
        st.title("👥 Person Management")
        st.markdown("---")
        
        st.info(f"Current Output Directory (from Image Processing): **{st.session_state.output_directory if st.session_state.output_directory else 'Not set'}**")

        if not st.session_state.output_directory:
            st.warning("Please set a valid Output Directory in the 'Image Processing' section first.")
        elif not os.path.isdir(st.session_state.output_directory):
            st.warning("The specified Output Directory does not exist. Please create it or set a valid path in 'Image Processing'.")
        
        if st.session_state.output_directory and os.path.isdir(st.session_state.output_directory):
            try:
                all_persons_data = db_manager.get_all_persons() 
                person_options = {f"{data.get('name_label', 'N/A')} ({person_id})": person_id for person_id, data in all_persons_data.items()}

                if person_options:
                    # --- Rename Person ---
                    st.header("Rename a Person")
                    
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        selected_person_key = st.selectbox(
                            "Select Person to Rename:", options=list(person_options.keys()), key="select_rename_person"
                        )
                        new_name = st.text_input("Enter New Name:", value="", key="new_name_input")

                        if st.button("Update Name"):
                            if not new_name:
                                st.error("New name cannot be empty.")
                            else:
                                person_id_to_rename = person_options[selected_person_key]
                                if update_person_name(person_id_to_rename, new_name, st.session_state.output_directory):
                                    st.success(f"Successfully renamed '{selected_person_key}' to '{new_name}'.")
                                    refresh_data()
                                    time.sleep(0.5)
                                    force_rerun()
                                else:
                                    st.error("Failed to rename person.")
                    
                    with col2:
                        st.write("**Preview**")
                        if selected_person_key:
                            person_id_to_preview = person_options[selected_person_key]
                            display_image_preview(person_id_to_preview, all_persons_data)

                    st.markdown("---")

                    # --- Merge Persons ---
                    st.header("Merge Persons")
                    
                    # --- UPDATED SECTION: MERGE PREVIEW ---
                    col1_merge, col2_merge = st.columns([2, 1])
                    
                    with col1_merge:
                        target_person_key = st.selectbox(
                            "Select Target Person (to merge into)", options=list(person_options.keys()), key="target_merge"
                        )
                        source_persons_keys = st.multiselect(
                            "Select Source Persons (to be merged)", options=list(person_options.keys()), key="source_merge"
                        )
                        
                        if st.button("Merge Persons"):
                            target_id = person_options[target_person_key]
                            source_ids = [person_options[key] for key in source_persons_keys]
                            
                            if target_id in source_ids:
                                st.error("Target person cannot be in the source list.")
                            elif not source_ids:
                                st.warning("Please select at least one source person.")
                            else:

                                if merge_persons(target_id, source_ids, st.session_state.output_directory):
                                    st.success("Successfully merged persons and their folders.")
                                    refresh_data()
                                    time.sleep(0.5)
                                    st.rerun()
                                else:
                                    st.error("Failed to merge persons and/or folders.")

                    with col2_merge:
                        st.write("**Target Preview**")
                        if target_person_key:
                            target_id_to_preview = person_options[target_person_key]
                            display_image_preview(target_id_to_preview, all_persons_data)
                        
                        st.write("**Sources Preview**")
                        if source_persons_keys:
                            for source_key in source_persons_keys:
                                source_id_to_preview = person_options[source_key]
                                display_image_preview(source_id_to_preview, all_persons_data)
                        else:
                            st.info("No sources selected to preview.")
                    # --- END OF UPDATED SECTION ---

                else:
                    st.info("No persons available to manage.")
            except Exception as e:
                st.error(f"Failed to load person management tools: {e}")


else:
    st.error("Database connection failed. Please check your MongoDB instance and configuration.")