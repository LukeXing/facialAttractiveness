import cv2
import math


def calculate_distance(pt1, pt2):
    """
    Calculates the Euclidean distance between two points.
    """
    return math.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)


def calculate_ratio(feature1, feature2):
    """
    Calculates the golden ratio between two facial features.
    """
    larger = max(feature1, feature2)
    smaller = min(feature1, feature2)
    return larger / smaller


def analyze_face(image_path):
    """
    Analyzes the facial features in an image using the golden ratio approach.
    """
    # Load the image and detect the face using OpenCV's face detection cascade classifier.
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        print('No face detected.')
        return

    # Assume the first detected face is the correct one and extract the facial features.
    (x, y, w, h) = faces[0]
    eye1_x = x + w // 4
    eye2_x = x + 3 * w // 4
    eye_y = y + h // 3
    nose_x = x + w // 2
    nose_y = y + 2 * h // 3
    mouth_x = x + w // 2
    mouth_y = y + 7 * h // 8

    # Calculate the distances and ratios between the facial features.
    eye_distance = calculate_distance((eye1_x, eye_y), (eye2_x, eye_y))
    nose_width = calculate_distance((eye1_x, eye_y), (eye2_x, eye_y))
    eye_nose_distance = calculate_distance((eye1_x, eye_y), (nose_x, nose_y))
    nose_mouth_distance = calculate_distance((nose_x, nose_y), (mouth_x, mouth_y))

    eye_nose_ratio = calculate_ratio(eye_distance, nose_width)
    nose_mouth_ratio = calculate_ratio(eye_nose_distance, nose_mouth_distance)

    # Calculate the total score for the face based on the ratios.
    total_score = eye_nose_ratio + nose_mouth_ratio

    # Output the results.
    print('Eye-to-eye distance to nose width ratio: {:.2f}'.format(eye_nose_ratio))
    print('Eye-to-nose distance to nose-to-mouth distance ratio: {:.2f}'.format(nose_mouth_ratio))
    print('Total score: {:.2f}'.format(total_score))


# Example usage:
analyze_face('facialAttractiveness/trainingData/prop 1.jpg')
