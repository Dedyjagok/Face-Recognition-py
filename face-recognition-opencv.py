import sys


import cv2
import os
import pickle
import numpy as np
from datetime import datetime
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

# Global variables
face_db = {}
face_db_file = "face_embeddings.pkl"

# Initialize models
print("Initializing PyTorch models...")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load models with correct parameters
mtcnn = MTCNN(
    image_size=160, margin=20, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, device=device,
    keep_all=True  # Keep all detected faces
)

# Face recognition model
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
print("Models loaded successfully")

def save_face_db():
    """Save face database to disk"""
    with open(face_db_file, 'wb') as f:
        pickle.dump(face_db, f)
    print(f"Face database saved with {len(face_db)} entries")

def load_face_db():
    """Load face database from disk if available"""
    global face_db
    if os.path.exists(face_db_file):
        try:
            with open(face_db_file, 'rb') as f:
                face_db = pickle.load(f)
            print(f"Loaded face database with {len(face_db)} entries")
            return True
        except Exception as e:
            print(f"Error loading face database: {e}")
    return False

def add_face_to_db(name, images_folder):
    """Add a person's face embeddings to the database from a folder of images"""
    global face_db
    face_embeddings = []
    
    # Check if folder exists
    if not os.path.exists(images_folder):
        print(f"Folder not found: {images_folder}")
        return False
    
    # Get list of image files
    valid_extensions = ('.png', '.jpg', '.jpeg')
    image_files = [f for f in os.listdir(images_folder) 
                   if f.lower().endswith(valid_extensions)]
    
    print(f"Processing {len(image_files)} images for {name}...")
    processed_count = 0
    
    # Process images in small batches to avoid memory issues
    batch_size = 5
    for i in range(0, len(image_files), batch_size):
        batch = image_files[i:i+batch_size]
        
        for filename in batch:
            try:
                # Full path to image
                img_path = os.path.join(images_folder, filename)
                
                # Use PIL to open image
                img = Image.open(img_path).convert('RGB')
                
                # Use the same detection approach as process_frame
                # First detect faces
                boxes, probs = mtcnn.detect(img)
                
                if boxes is not None and len(boxes) > 0:
                    # Extract aligned faces
                    faces = mtcnn.extract(img, boxes, save_path=None)
                    
                    if faces is not None and len(faces) > 0:
                        # Get embedding from the first detected face
                        face_tensor = faces[0]
                        if face_tensor is not None:
                            with torch.no_grad():
                                embedding = resnet(face_tensor.unsqueeze(0).to(device))
                            
                            # Store embedding
                            face_embeddings.append(embedding.cpu().numpy().flatten())
                            processed_count += 1
                            print(f"Processed {filename} - face detected")
                        else:
                            print(f"No valid face tensor in {filename}")
                    else:
                        print(f"No valid face extracted in {filename}")
                else:
                    print(f"No face detected in {filename}")
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    print(f"Successfully processed {processed_count} images with faces")
    
    if face_embeddings:
        # Store the average embedding for this person
        face_db[name] = np.mean(face_embeddings, axis=0)
        save_face_db()
        return True
    else:
        print(f"No valid faces found for {name}")
        return False

def recognize_face(face_embedding):
    """Match a face embedding to the database using cosine similarity"""
    if not face_db:
        return "Unknown", 0
        
    max_similarity = -1
    recognized_name = "Unknown"
    
    for name, db_embedding in face_db.items():
        # Calculate cosine similarity
        similarity = np.dot(face_embedding, db_embedding) / (np.linalg.norm(face_embedding) * np.linalg.norm(db_embedding))
        if similarity > max_similarity:
            max_similarity = similarity
            recognized_name = name
    
    # Threshold for recognition
    threshold = 0.60
    if max_similarity < threshold:
        return "Unknown", max_similarity
    return recognized_name, max_similarity

# Add these global variables at the top of your file with other globals
last_detection_boxes = None
last_detection_names = None
last_detection_confidences = None
last_detection_time = 0
detection_timeout = 1.0  # How long to display detection results in seconds

def process_frame(frame, frame_count=0):
    """Process a video frame for face detection and recognition"""
    global last_detection_boxes, last_detection_names, last_detection_confidences, last_detection_time
    
    # Copy frame for display regardless of processing
    display = frame.copy()
    
    # Add timestamp to every frame for consistency
    current_time = datetime.now()
    timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(display, timestamp, 
               (10, display.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Check if we should do detection on this frame
    do_detection = (frame_count % 3 == 0)
    
    # Clear old detections if they timeout
    if (datetime.now() - datetime.fromtimestamp(last_detection_time)).total_seconds() > detection_timeout:
        last_detection_boxes = None
    
    if do_detection:

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        
        try:

            batch_boxes, probs, points = mtcnn.detect(pil_img, landmarks=True)
            batch_faces = mtcnn.extract(pil_img, batch_boxes, save_path=None)
            
            # Process each detected face and update cache
            if batch_boxes is not None and batch_faces is not None:

                last_detection_boxes = []
                last_detection_names = []
                last_detection_confidences = []
                
                for i, (box, face_tensor) in enumerate(zip(batch_boxes, batch_faces)):
                    if face_tensor is None:
                        continue
                    

                    with torch.no_grad():
                        embedding = resnet(face_tensor.unsqueeze(0).to(device))
                    

                    name, confidence = recognize_face(embedding.cpu().numpy().flatten())
                    

                    last_detection_boxes.append([int(b) for b in box])
                    last_detection_names.append(name)
                    last_detection_confidences.append(confidence)
                
                # Update timestamp
                last_detection_time = current_time.timestamp()
        
        except Exception as e:
            print(f"Error in face detection/recognition: {e}")
    
    # Always draw the cached detections
    if last_detection_boxes:
        for i, box in enumerate(last_detection_boxes):
            name = last_detection_names[i]
            confidence = last_detection_confidences[i]
            
            # Choose color based on recognition status
            if name == "Unknown":
                border_color = (0, 0, 255)  # Red for unknown faces
            else:
                border_color = (0, 255, 0)  # Green for recognized faces
            
            # Draw rectangle with appropriate color
            cv2.rectangle(display, (box[0], box[1]), (box[2], box[3]), border_color, 2)
            
            # Draw name and confidence
            label = f"{name} ({confidence:.2f})"
            cv2.putText(display, label, (box[0], box[1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, border_color, 2)
    
    return display

def capture_reference_images(cap, name, num_images=5):
    """Capture reference images for a new person"""
    # Create folder if it doesn't exist
    folder = f"reference_faces/{name.lower()}"
    os.makedirs(folder, exist_ok=True)
    
    print(f"Taking {num_images} photos for {name}...")
    count = 0
    
    while count < num_images:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Error reading from camera")
            break
        
        # Convert to RGB (PIL format)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        
        # Detect faces
        try:
            # Detect faces
            boxes, probs = mtcnn.detect(pil_img)
            
            display = frame.copy()
            
            if boxes is not None and len(boxes) > 0:
                # Show all faces
                for i, (box, prob) in enumerate(zip(boxes, probs)):
                    if prob < 0.9:  # High confidence threshold
                        continue
                        
                    # Convert to integers
                    box = [int(b) for b in box]
                    x1, y1, x2, y2 = box
                    
                    # Check face size
                    face_width = x2 - x1
                    face_height = y2 - y1
                    
                    if face_width > 100 and face_height > 100:  # Good size face
                        # Save image
                        filename = f"{folder}/{name}_{count+1}.jpg"
                        cv2.imwrite(filename, frame)
                        count += 1
                        
                        # Show feedback
                        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(display, f"Captured {count}/{num_images}", (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        print(f"Saved {filename}")
                        cv2.waitKey(500)  # Wait half a second between captures
                        break  # Only save one face per image
                    else:
                        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        cv2.putText(display, "Move closer", (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(display, "No face detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        except Exception as e:
            print(f"Error during capture: {e}")
            cv2.putText(display, f"Error: {str(e)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('Capturing Reference Images', display)
        
        # Wait for key or delay
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cv2.destroyWindow('Capturing Reference Images')
    return count == num_images

def main():
    # Try to load existing database
    load_face_db()
    
    # Set up camera
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Check for reference faces
    if len(face_db) == 0:
        print("No faces in database.")
        print("You'll need to add faces to the recognition database.")
        print("Press 'a' during runtime to add a new person.")
        
        # Optional: Automatically prompt to add the first user
        add_first_user = input("Would you like to add a face now? (y/n): ").lower().strip() == 'y'
        if add_first_user:
            name = input("Enter name for the first person: ")
            if name:
                if os.path.exists("images"):
                    # If images folder exists, ask if they want to use it
                    use_images = input(f"Use existing images in 'images' folder for {name}? (y/n): ").lower().strip() == 'y'
                    if use_images:
                        add_face_to_db(name, "images")
                    else:
                        if capture_reference_images(cap, name):
                            add_face_to_db(name, f"reference_faces/{name.lower()}")
                else:
                    # Capture new reference images
                    print("Let's capture some reference images.")
                    if capture_reference_images(cap, name):
                        add_face_to_db(name, f"reference_faces/{name.lower()}")
    
    print("Starting face recognition...")
    print("Press 'q' to quit, 'a' to add a new person, 's' to save the database")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading from camera")
            break
        
        frame_count += 1
        processed = process_frame(frame, frame_count)
        
        cv2.imshow('Face Recognition', processed)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):
            name = input("Enter name for the new person: ")
            if name:
                if capture_reference_images(cap, name):
                    add_face_to_db(name, f"reference_faces/{name.lower()}")
        elif key == ord('s'):
            save_face_db()
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()