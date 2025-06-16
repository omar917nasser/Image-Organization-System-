# Project Title: AI-Powered Face Recognition and Image Organizer

## 📖 Overview

This project is a sophisticated, AI-driven application designed to automatically organize a collection of images based on the faces of individuals detected within them. It leverages state-of-the-art machine learning models for highly accurate face detection and recognition, coupled with a robust database system for storing and managing facial data. The application is user-friendly, featuring a Streamlit-based graphical user interface (GUI) for easy interaction, allowing users to process image libraries, manage identified individuals, and maintain a cleanly organized photo collection.

The system is built with a modular architecture, separating concerns into distinct components for AI modeling, database interactions, core face processing logic, file organization, and the user interface. This design ensures scalability, maintainability, and ease of future development.

## ✨ Key Features

* **Advanced Face Detection:** Utilizes the powerful YOLO (You Only Look Once) model to accurately detect faces within images, even in challenging conditions.
* **High-Accuracy Face Recognition:** Employs the FaceNet model, pre-trained on the extensive VGGFace2 dataset, to generate unique facial embeddings for precise identification.
* **Automated Image Organization:** Automatically sorts images into folders named after the identified individuals. Images with multiple recognized faces are copied to each person's respective folder.
* **Intelligent Person Management:**
    * **New Person Detection:** Automatically creates new person profiles for unrecognized faces.
    * **Person Renaming:** Allows users to assign or update names for identified individuals, which also renames the corresponding folders on disk.
    * **Person Merging:** Provides functionality to merge duplicate entries of the same person, consolidating their images and facial data.
* **Robust Data Persistence:** Leverages MongoDB to store facial embeddings, person information, and image paths, ensuring data integrity and efficient retrieval.
* **User-Friendly Interface:** A comprehensive Streamlit GUI provides a seamless user experience for:
    * **Dashboard:** An overview of the number of people and images managed.
    * **Image Processing:** A simple interface to specify input and output directories to start the organization process.
    * **Person Management:** Tools to rename and merge individuals with visual feedback.
* **Efficient Processing:** The application is optimized to run on a CUDA-enabled GPU for accelerated AI model inference, with a fallback to CPU if a GPU is not available.

## 🛠️ Tech Stack & Dependencies

* **Python 3.x**
* **AI / Machine Learning:**
    * `ultralytics` (for YOLO face detection)
    * `facenet-pytorch` (for FaceNet face recognition)
    * `torch` & `torchvision` (the core deep learning framework)
* **Database:**
    * `pymongo` (for interacting with MongoDB)
* **GUI:**
    * `streamlit`
* **Image Processing & File Handling:**
    * `opencv-python`
    * `Pillow` (PIL)
    * `numpy`

## 🚀 Getting Started

### Prerequisites

* Python 3.8 or higher
* An running instance of MongoDB. You can run this locally or use a cloud-based service like MongoDB Atlas.
* The YOLOv11l-face model file (`yolov11l-face.pt`) should be placed in the project's root directory.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```

2.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You will need to create a `requirements.txt` file listing all the dependencies mentioned in the "Tech Stack & Dependencies" section.)*

### Configuration

The application's configuration is managed in the `config.py` file. You can modify the following parameters:

* **MongoDB Connection:**
    * `connection_uri`: The connection string for your MongoDB instance.
    * `database_name`: The name of the database to be used.
* **Face Recognition Thresholds:**
    * `similarity_threshold`: The cosine similarity score above which a face is considered a match to an existing person (default: 0.8).
    * `confidence_threshold`: The confidence score above which a detected object is considered a face by the YOLO model (default: 0.25).

### Running the Application

To start the Streamlit-based user interface, run the following command in your terminal:

```bash
streamlit run gui.py
```

This will open the application in your default web browser.

## 💻 How to Use

1.  **Navigate to the "Image Processing" page** using the sidebar.
2.  **Enter the path to the directory containing the images** you want to organize in the "Input Directory Path" field.
3.  **Enter the path to the directory where you want the organized folders to be created** in the "Output Directory Path" field.
4.  **Click the "Start Processing" button.** The application will then:
    * Scan each image for faces.
    * Generate a facial embedding for each detected face.
    * Compare the embedding with the known individuals in the database.
    * If a match is found, the image will be copied to that person's folder.
    * If no match is found, a new person profile will be created, and the image will be placed in a new folder for that person.
    * Images with no faces will be moved to a `_no_faces` directory.
5.  **Navigate to the "Person Management" page** to:
    * **Rename a Person:** Select a person from the dropdown, enter their new name, and click "Update Name." This will update their name in the database and rename their corresponding folder.
    * **Merge Persons:** Select a target person (who will remain) and one or more source people (who will be merged into the target). Click "Merge Persons" to combine their data and images.
6.  **The "Dashboard" page** provides a quick overview of the total number of people and images that the application is currently managing.

## 📂 Project Structure

```
.
├── aiModels.py             # Initializes and loads the YOLO and FaceNet models.
├── config.py               # Stores configuration variables for the application.
├── databaseManager.py      # Handles all interactions with the MongoDB database.
├── faceProcessing.py       # Contains the core logic for face detection, embedding generation, and person identification.
├── fileOrganizer.py        # Manages the process of reading images and organizing them into folders.
├── folderSync.py           # Synchronizes folder names and structures with the database.
├── gui.py                  # The main Streamlit application for the user interface.
├── yolov11l-face.pt        # The pre-trained YOLO model for face detection (must be downloaded).
└── requirements.txt        # A list of all python dependencies.
```

## Contact
You can contanct me at omar.nasser.pro@gmail.com 
For any information about the project
