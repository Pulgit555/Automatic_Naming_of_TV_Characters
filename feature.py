import cv2
import dlib
import matplotlib.pyplot as plt

# give input path of the image
def features(path_img):
    img = cv2.imread(path_img)
    detector = dlib.get_frontal_face_detector()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    points = [17, 21, 22, 26, 30, 31, 35, 48, 54]
    x_y_coordinates = []
    features = []
    for face in faces:
        x1 = face.left() # left point
        y1 = face.top() # top point
        x2 = face.right() # right point
        y2 = face.bottom() # bottom point
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        x_y_coordinates.append([x1,y1,x2,y2])
        landmarks = predictor(gray, face)
        face_feature = []
        for point in points:
            x = landmarks.part(point).x
            y = landmarks.part(point).y
            img = cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
            face_feature.append([x,y])
        features.append(face_feature)
        # returns highlighted image, 9 features for each face, bounding box coordinates for each face
    return img, features, x_y_coordinates
