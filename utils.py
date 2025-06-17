import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def initialize_hand_detector(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
    """Initialize the MediaPipe hand detector"""
    return mp_hands.Hands(
        static_image_mode=static_image_mode,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence
    )

def detect_hands(image, hands):
    """Detect hands in an image and return the processed results"""
    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image and detect hands
    results = hands.process(image_rgb)
    
    return results

def extract_hand_landmarks(results):
    """Extract hand landmarks from the detection results"""
    landmarks = []
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract x, y coordinates of each landmark
            hand_points = []
            for landmark in hand_landmarks.landmark:
                hand_points.append([landmark.x, landmark.y, landmark.z])
            
            landmarks.append(hand_points)
    
    return landmarks

def draw_hand_landmarks(image, results):
    """Draw hand landmarks on the image"""
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
    
    return image

def preprocess_landmarks(landmarks):
    """Preprocess landmarks for model input"""
    if not landmarks:
        return np.zeros((1, 21 * 3))  # Return zeros if no landmarks detected
    
    # Take the first hand detected
    hand = landmarks[0]
    
    # Flatten the landmarks into a 1D array
    flat_landmarks = np.array(hand).flatten()
    
    # Normalize the landmarks
    x_min, y_min = min([p[0] for p in hand]), min([p[1] for p in hand])
    x_max, y_max = max([p[0] for p in hand]), max([p[1] for p in hand])
    
    for i in range(0, len(flat_landmarks), 3):
        if x_max > x_min:
            flat_landmarks[i] = (flat_landmarks[i] - x_min) / (x_max - x_min)
        if y_max > y_min:
            flat_landmarks[i+1] = (flat_landmarks[i+1] - y_min) / (y_max - y_min)
    
    return flat_landmarks.reshape(1, -1)