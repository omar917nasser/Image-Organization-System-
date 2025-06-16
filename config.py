DEFAULT_CONFIG = {
    "mongodb": {
        "connection_uri": "mongodb://localhost:27017/",
        "database_name": "imageProject"
    },
    "face_recognition": {
        "similarity_threshold": 0.8,
        "confidence_threshold": 0.25
    }
}

CONNECTION_URI = DEFAULT_CONFIG['mongodb']['connection_uri']
DATABASE_NAME = DEFAULT_CONFIG['mongodb']['database_name']
SIMILARITY_THRESHOLD = DEFAULT_CONFIG['face_recognition']['similarity_threshold']
CONFIDENCE_THRESHOLD = DEFAULT_CONFIG['face_recognition']['confidence_threshold']