from tensorflow.keras.models import load_model  # Importing the load_model function from TensorFlow Keras
from time import sleep  # Importing the sleep function from the time module
from tensorflow.keras.preprocessing.image import img_to_array  # Importing img_to_array function from TensorFlow Keras
from tensorflow.keras.preprocessing import image  # Importing the image module from TensorFlow Keras preprocessing
import cv2  # Importing the OpenCV library for computer vision tasks
import numpy as np  # Importing the NumPy library for numerical operations

# Loading the Haar cascade XML file for face detection
face_classifier = cv2.CascadeClassifier(r'C:\Users\sayan\Desktop\New folder\haarcascade_frontalface_default.xml')

# Loading the pre-trained model for emotion detection
classifier = load_model(r'C:\Users\sayan\Desktop\New folder\model.h5')

# List of emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Opening the video capture device
cap = cv2.VideoCapture(0)

while True:
    # Capturing the video frame
    _, frame = cap.read()

    labels = []

    # Converting the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecting faces in the grayscale frame
    faces = face_classifier.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        # Drawing a rectangle around each detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # Extracting the region of interest (ROI) from the grayscale frame
        roi_gray = gray[y:y + h, x:x + w]

        # Resizing the ROI to match the input size of the model
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            # Preprocessing the ROI by scaling and converting to array
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # Predicting the emotion label using the loaded model
            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]

            label_position = (x, y)
            # Displaying the predicted emotion label on the frame
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            # Displaying "No Faces" text on the frame if no face is detected
            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Displaying the frame with emotion labels
    cv2.imshow('Emotion Detector', frame)

    # Breaking the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Releasing the video capture device and closing all windows
cap.release()
cv2.destroyAllWindows()
