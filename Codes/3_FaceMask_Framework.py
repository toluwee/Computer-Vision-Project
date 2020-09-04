import cv2
import numpy as np
import dlib
from math import hypot
import os
arr = os.listdir('D:/Deep learning/data_test/test1/')

for i in arr:
# Loading Camera and Nose image and Creating mask
    print(i)
    cap = cv2.imread(i)
    nose_image = cv2.imread("facemaskimage2.png")

#_, frame = cap.read()
    cols, rows, channels = np.shape(cap)
    nose_mask = np.zeros((rows, cols), np.uint8)

# Loading Face detector
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


#nose_mask.fill(0)
    gray_frame = cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)
    faces = detector(cap)
    for face in faces:
        landmarks = predictor(gray_frame, face)

        center_nose = (landmarks.part(30).x, landmarks.part(30).y)
        left_nose = (landmarks.part(1).x, landmarks.part(1).y)
        right_nose = (landmarks.part(26).x, landmarks.part(26).y)
        bottom_nose=(landmarks.part(8).x, landmarks.part(8).y)
        nose_width = int(hypot(left_nose[0] - right_nose[0],
                           left_nose[1] - right_nose[1])*1.05)
        nose_height = int(nose_width*0.7)


        # New nose position
        top_left = (int(left_nose[0]),int(left_nose[1]))
        bottom_right = (int(right_nose[0]),int(bottom_nose[1]))
        #cv2.rectangle(frame,(int(left_nose[0]),int(left_nose[1])),(int(right_nose[0]),int(bottom_nose[1])),(0,255,0),2)


        # Adding the new nose
        nose_pig = cv2.resize(nose_image, (nose_width, nose_height))
        nose_pig_gray = cv2.cvtColor(nose_pig, cv2.COLOR_BGR2GRAY)
        _, nose_mask = cv2.threshold(nose_pig_gray, 25, 255, cv2.THRESH_BINARY_INV)
        nose_area = cap[top_left[1]: top_left[1] + nose_height,
                top_left[0]: top_left[0] + nose_width]

        try:
            nose_area_no_nose = cv2.bitwise_and(nose_area, nose_area, mask=nose_mask)
            final_nose = cv2.add(nose_area_no_nose, nose_pig)
            cap[top_left[1]: top_left[1] + nose_height,
                    top_left[0]: top_left[0] + nose_width] = final_nose

            filename = i+'_1.jpg'
            cv2.imwrite(filename, cap)
        except:
            pass

        #cv2.imshow('image3',cap)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
