import cv2
import mediapipe as mp
import time
import numpy as np

class handDetector():
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity, self.detectionCon,
                                        self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def getResults(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

    def findHands(self, img, draw=True):
        self.getResults(img)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNum=0, draw=True, coord = "Absolute"):
        self.getResults(img)

        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNum]
            h, w, c = img.shape

            for id, lm in enumerate(myHand.landmark):
                if coord == "Relative":
                    # print the id and the landmark location
                    # convert to pixels
                    cx, cy, z = int(lm.x * w), int(lm.y * h), lm.z
                    lmList.append([id, cx, cy, z])
                else:
                    lmList.append([id, lm.x, lm.y, lm.z])

                if draw:
                    cv2.circle(img, (int(lm.x * w), int(lm.y * h)), 15, (255, 0, 0), cv2.FILLED)

        if coord == "Normalized" and lmList:
            n_factor = ((((lmList[0][1]-lmList[5][1])**2)+((lmList[0][2]-lmList[5][2])**2)+((lmList[0][3]-lmList[5][3])**2))**0.5)
            wrist = lmList[0]
            for list in lmList:
                list[1:] = np.divide(np.array(list[1:]) - np.array(wrist[1:]),n_factor)
            return lmList
        return lmList


def main():
    current_time = 0
    previous_time = 0
    cap = cv2.VideoCapture(0)

    detector = handDetector()
    while cap.isOpened():
        success, img = cap.read()
        detector.findHands(img)
        lmList = detector.findPosition(img, draw=False, coord="Normalized")
        if len(lmList):
            print(lmList)

        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)

        cv2.imshow("Image", img)
        # pressing q will shut down the video feed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
