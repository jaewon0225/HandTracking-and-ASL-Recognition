import cv2
import csv
import os
import HandTrackingModule as htm

detector = htm.handDetector()

directory = 'archive/asl_alphabet_train/asl_alphabet_train'
for folder in os.listdir(directory):
    if os.path.isdir(os.path.join(directory,folder)):
        joinedCoords = []
        print(folder)
        for filename in os.listdir(os.path.join(directory,folder)):
            img = cv2.imread(os.path.join(directory,folder,filename))
            LMList = detector.findPosition(img, draw=False, coord="Normalized")
            rows = []
            if LMList:
                for lm in LMList[1:]:
                    rows += lm[1:]
                joinedCoords.append(list(rows))

        with open(os.path.join("processed_data","{}.csv".format(folder)), 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            # write multiple rows
            writer.writerows(joinedCoords)
            print("saved")