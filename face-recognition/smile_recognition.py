import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)   
        roi_gray = gray[y: y+ int(h/1.5), x:x+w]
        roi_color = frame[y:y+ int(h/1.5), x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 22)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        
        smile_gray = gray[y+int(h/1.5): y+h, x:x+w]
        smile_color = frame[y+int(h/1.5):y+h, x:x+w]
        smiles = smile_cascade.detectMultiScale(smile_gray, 1.1, 22) 
       
        for (sx, sy, sw, sh) in smiles :
            cv2.rectangle(smile_color, (sx,ey-int(h/1.5)), (sx+sw, sy+sh), (180, 255, 0), 2)
    return frame



video_capture = cv2.VideoCapture(0)
video_capture.open(0)
while(video_capture.isOpened()):
    ret, frame = video_capture.read()
    
    if(ret):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        canvas = detect (gray, frame)
        cv2.imshow('Video', canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
video_capture.release()
cv2.destroyAllWindows()
        