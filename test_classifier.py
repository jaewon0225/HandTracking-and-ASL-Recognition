import pickle
import HandTrackingModule as hp
import cv2
import time

# load the model from disk
loaded_model = pickle.load(open('knnpickle_file', 'rb'))

current_time = 0
previous_time = 0
cap = cv2.VideoCapture(0)

detector = hp.handDetector()
while cap.isOpened():
    success, img = cap.read()
    detector.findHands(img)
    LMList = detector.findPosition(img, draw=False, coord="Normalized")
    input = []
    rows = []
    if LMList:
        for lm in LMList[1:]:
            rows += lm[1:]
        input.append(rows)
        prediction = loaded_model.predict(input)
        print(prediction)
        cv2.putText(img, str(prediction[0]), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)

    cv2.imshow("Image", img)
    # pressing q will shut down the video feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
