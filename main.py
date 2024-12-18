import cv2

camera = cv2.VideoCapture(0)

face_path = "haarcascade_frontalface_default.xml"
eye_path = "haarcascade_eye.xml"

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + face_path)
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + eye_path)


while camera.isOpened():
    access , frame = camera.read()
    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    faces_detected = face_cascade.detectMultiScale(frame , 1.2 , 5)
    
    for x , y , h , w in faces_detected :
        roi_gary = gray[x:x+h , y:y+w]
        roi_color = frame[x:x+h , y:y+w]
        cv2.rectangle(frame , (x,y) , (x+h , y+w) , (0,255,0) , 2)
        eyes_detected = eye_cascade.detectMultiScale(roi_color)

        for ex , ey , height , width in eyes_detected:
            cv2.rectangle(roi_color , (ex,ey) , (ex+height , ey+width) , (255,0,0) , 2)

    cv2.imshow("window" , frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

cv2.destroyAllWindows()
camera.release()
