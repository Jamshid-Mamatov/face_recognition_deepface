import numpy 
import cv2

from detector import detect_face
cap=cv2.VideoCapture("ping_pong.mp4")


while cap.isOpened():
    ret,frame=cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    boxes=detect_face(frame)
    for box in boxes:
        cv2.rectangle(frame,(int(box[0]), int(box[1])),(int(box[2]), int(box[3])),(0, 0, 255),thickness=2)

                    # Show probability
        cv2.putText(frame, str("team member"), (int(box[2]), int(box[3])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

      
        cv2.imshow("frame",frame)
    if cv2.waitKey(1)==ord("q"):
        break
cap.release()
cv2.destroyAllWindows()