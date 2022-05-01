import cv2 as cv
#from cv2 import threshold
#from cv2 import CV_16S
import numpy as np

body_parts ={ "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

body_pairs = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

image_width = 600
image_height = 600

net = cv.dnn.readNetFromTensorflow("graph_opt.pb")
threshold = 0.2

img = cv.imread('./image.jpg', cv.IMREAD_UNCHANGED)

photo_height = img.shape[0]
photo_width = img.shape[1]
net.setInput(cv.dnn.blobFromImage(img, 1.0, (image_width, image_height), (127.5, 127.5, 127.5), swapRB=True, crop=False))

out = net.forward()
out = out[:, :19, :, :] #slicing output

assert(len(body_parts) == out.shape[1])

keyPoints = []
for i in range(len(body_parts)):
    a = out[0,i,:,:]
    _, conf, _, point = cv.minMaxLoc(a)
    x = (photo_width * keyPoints[0]) / out.shape[3]
    y = (photo_height * keyPoints[1]) / out.shape[2]
    keyPoints.append((int(x), int(y)) if conf > threshold else None)


for pair in body_pairs:
    partFrom = pair[0]
    partTo = pair[1]
    assert(partFrom in body_parts)
    assert(partTo in body_parts)

    idFrom = body_parts[partFrom]
    idTo = body_parts[partTo]

    if keyPoints[idFrom] and keyPoints[idTo]:
        cv.line(img, keyPoints[idFrom], keyPoints[idTo], (0, 255, 0), 3)
        cv.ellipse(img, keyPoints[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
        cv.ellipse(img, keyPoints[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

t, _ = net.getPerfProfile()

cv.imshow("test",img)

cv.destroyAllWindows() 