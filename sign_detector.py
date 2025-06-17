import cv2
import numpy as np
import pickle
import tensorflow as tf
from utils import initialize_hand_detector, detect_hands, extract_hand_landmarks, draw_hand_landmarks, preprocess_landmarks
import time

def load_model(model_type="rf"):
    """Load the trained model and labels"""
    if model_type == "rf":
        # Load Random Forest model
        with open('sign_language_rf_model.pkl', 'rb') as f:
            model = pickle.load(f)
    else:
        # Load Neural Network model
        model = tf.keras.models.load_model('sign_language_nn_model.h5')
    
    # Load labels
    with open('sign_language_labels.pkl', 'rb') as f:
        labels = pickle.load(f)
    
    return model, labels

def predict_sign(model, landmarks, model_type="rf"):
    """Predict the sign from hand landmarks"""
    # Preprocess landmarks
    processed_landmarks = preprocess_landmarks(landmarks)
    
    if model_type == "rf":
        # Predict using Random Forest
        prediction = model.predict(processed_landmarks)[0]
    else:
        # Predict using Neural Network
        prediction = np.argmax(model.predict(processed_landmarks))
    
    return prediction

def run_sign_detector(model_type="rf"):
    """Run the sign language detector"""
    # Load the model and labels
    model, labels = load_model(model_type)
    
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    
    # Initialize the hand detector
    hands = initialize_hand_detector()
