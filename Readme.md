# Face Recognition System

This project implements a real-time face recognition system using OpenCV and PyTorch. It uses MTCNN for face detection and InceptionResnetV1 for facial recognition, providing a simple but effective way to recognize faces from your webcam feed.

## Features

* Real-time face detection and recognition
* Support for multiple faces in the same frame
* Simple interface to add new people to the recognition database
* Color-coded recognition (green for known faces, red for unknown)
* Confidence scores displayed with each recognition

## Requirements

* Python 3.7+
* OpenCV
* PyTorch
* facenet-pytorch
* NumPy
* Pillow (PIL)

## Installation

1.  Clone this repository:
    ```bash
    git clone (https://github.com/Dedyjagok/Face-Recognition-py.git)
    cd Face-Recognition
    ```
2.  Install the required packages:
    ```bash
    # Option 1: Install using requirements.txt (recommended)
    pip install -r requirements.txt
    
    # Option 2: Install packages individually
    pip install opencv-python torch torchvision facenet-pytorch numpy pillow
    ```
3.  Make sure you have a working webcam connected to your computer.

## Usage

Running the Application

Run the main script:
```bash
python face-recognition-opencv.py
```

## Key Controls

* Press q to quit the application
* Press a to add a new person to the face database
* Press s to manually save the face database

## Adding New Faces

There are three ways to add faces to the recognition database:

**1. During runtime:**
* Press `a` while the program is running
* Enter the person's name when prompted
* The system will capture 5 reference images
* Stay still and face the camera during capture

**2. Using the images folder:**
* Create a folder named `images` in the project directory
* Add clear photos of the person's face to this folder
* The system will automatically use these on startup if no database exists

**3. Programmatically:**
* Call the `add_face_to_db(name, folder_path)` function where:
    * `name` is the person's name
    * `folder_path` is the path to a folder containing their face images

## Project Structure

* `face-recognition-opencv.py` - Main application file
* `face_embeddings.pkl` - Database of face embeddings (created automatically)
* `images` - Directory for reference face images
* `reference_faces/` - Directory where newly captured faces are stored
* `processed_images` - Directory for processed face images

## How It Works

1.  The system uses MTCNN to detect faces in the webcam feed
2.  Each detected face is passed through InceptionResnetV1 to create a 512-dimensional embedding
3.  These embeddings are compared to known faces in the database using cosine similarity
4.  If a match is found (similarity above 0.6), the person is recognized and their name is displayed

## Privacy and Security

* Face embeddings are stored locally in `face_embeddings.pkl`
* No facial data is transmitted over the internet
* Consider the privacy implications before posting images of others

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License and Third-Party Libraries

Copyright (c) 2023 Dedy  
All rights reserved.

This project and its contents are protected under copyright law. No part of this project 
may be reproduced, distributed, or transmitted in any form or by any means without the 
prior written permission of the copyright holder.

### Third-Party Libraries

This project uses several open-source libraries, each with their own licenses:

- **OpenCV**: BSD 3-Clause License
- **PyTorch**: BSD-style license
- **facenet-pytorch**: MIT License
- **NumPy**: BSD 3-Clause License
- **Pillow**: HPND License (Historical Permission Notice and Disclaimer)

While these libraries are open source, my implementation and code remain under All Rights Reserved 
copyright. The inclusion of these libraries does not grant permission to use this software beyond 
what is explicitly authorized by the copyright holder.

For permission to use this software, please contact dedyhutahaean2005@gmail.com.