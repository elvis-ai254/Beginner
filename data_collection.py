import os
import cv2
import numpy as np
import time
from utils import initialize_hand_detector, detect_hands, extract_hand_landmarks, draw_hand_landmarks

def collect_data():
    # Create data directory if it doesn't exist
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    
    # Initialize the hand detector
    hands = initialize_hand_detector()
    
    # Define the signs to collect
    signs = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    
    # Number of samples per sign
    num_samples = 100
    
    for sign in signs:
        # Create directory for this sign if it doesn't exist
        sign_dir = os.path.join(data_dir, sign)
        if not os.path.exists(sign_dir):
            os.makedirs(sign_dir)
        
        print(f"Collecting data for sign: {sign}")
        print(f"Press 's' to start collecting {num_samples} samples")
        
        # Wait for user to press 's' to start collecting
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Display instructions
            cv2.putText(frame, f"Prepare to show sign '{sign}'", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press 's' to start collecting", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Data Collection', frame)
            
            key = cv2.waitKey(1)
            if key == ord('s'):
                break
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
        
        # Start collecting samples
        count = 0
        while count < num_samples:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Detect hands
            results = detect_hands(frame, hands)
            
            # Extract landmarks
            landmarks = extract_hand_landmarks(results)
            
            if landmarks:
                # Save landmarks
                np.save(os.path.join(sign_dir, f"{count}.npy"), landmarks[0])
                
                # Draw landmarks on the frame
                frame = draw_hand_landmarks(frame, results)
                
                count += 1
                print(f"Collected sample {count}/{num_samples} for sign {sign}")
            
            # Display progress
            cv2.putText(frame, f"Collecting sign '{sign}': {count}/{num_samples}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Data Collection', frame)
            
            # Add a small delay to avoid collecting too many similar frames
            time.sleep(0.1)
            
            if cv2.waitKey(1) == ord('q'):
                break
        
        print(f"Completed collecting data for sign {sign}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("Data collection completed!")

if __name__ == "__main__":
    collect_data()