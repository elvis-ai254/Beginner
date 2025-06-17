import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

def load_data(data_dir="data"):
    """Load the collected data and prepare it for training"""
    X = []
    y = []
    
    # Get all sign directories
    sign_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    for i, sign in enumerate(sign_dirs):
        sign_path = os.path.join(data_dir, sign)
        
        # Get all sample files
        sample_files = [f for f in os.listdir(sign_path) if f.endswith('.npy')]
        
        for sample_file in sample_files:
            # Load the landmarks
            landmarks = np.load(os.path.join(sign_path, sample_file))
            
            # Flatten the landmarks
            flat_landmarks = landmarks.flatten()
            
            X.append(flat_landmarks)
            y.append(i)  # Use index as the label
    
    return np.array(X), np.array(y), sign_dirs

def train_random_forest_model(X, y):
    """Train a Random Forest classifier"""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Random Forest Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    
    return model

def train_neural_network(X, y, num_classes):
    """Train a neural network model"""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert labels to one-hot encoding
    y_train_one_hot = to_categorical(y_train, num_classes=num_classes)
    y_test_one_hot = to_categorical(y_test, num_classes=num_classes)
    
    # Create the model
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model
    history = model.fit(
        X_train, y_train_one_hot,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test_one_hot),
        verbose=1
    )
    
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test_one_hot)
    print(f"Neural Network Accuracy: {accuracy:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    
    return model

def save_model(model, model_type, sign_labels):
    """Save the trained model and labels"""
    if model_type == "rf":
        with open('sign_language_rf_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        with open('sign_language_labels.pkl', 'wb') as f:
            pickle.dump(sign_labels, f)
        print("Random Forest model saved as 'sign_language_rf_model.pkl'")
    else:
        model.save('sign_language_nn_model.h5')
        with open('sign_language_labels.pkl', 'wb') as f:
            pickle.dump(sign_labels, f)
        print("Neural Network model saved as 'sign_language_nn_model.h5'")

if __name__ == "__main__":
    # Load the data
    X, y, sign_labels = load_data()
    
    print(f"Loaded {len(X)} samples for {len(sign_labels)} signs")
    print(f"Signs: {sign_labels}")
    
    # Train Random Forest model
    rf_model = train_random_forest_model(X, y)
    save_model(rf_model, "rf", sign_labels)
    
    # Train Neural Network model
    nn_model = train_neural_network(X, y, len(sign_labels))
    save_model(nn_model, "nn", sign_labels)