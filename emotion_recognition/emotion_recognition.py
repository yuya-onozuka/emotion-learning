import os
import cv2
import requests

headers = {
    # Request headers
    'Content-Type': 'application/octet-stream',
    'Ocp-Apim-Subscription-Key': '760f908d0ed34f7188ee667ce9a316a0',
}

write_file_name = './tmp.jpg'
api_url = 'https://emotion-recognition.cognitiveservices.azure.com/face/v1.0/detect'

cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
writer = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))

params = {
    # Request parameters
    'returnFaceId': 'false',
    'returnFaceLandmarks': 'false',
    'returnFaceAttributes': 'emotion',
}

count = 1
try:
    while True:
        ret, frame = cap.read()
        cv2.imwrite(write_file_name, frame)
        with open(write_file_name, 'rb') as f:
            img = f.read()
        response = requests.post(api_url,params=params,headers=headers,data=img)

        cv2.imshow('Raw Frame', frame)

        data = response.json()

        for id in range(len(data)):
            rect = data[id]['faceRectangle']
            emotion = data[id]['faceAttributes']['emotion']

            top = int(rect['top'])
            left = int(rect['left'])
            btm = top + int(rect['height'])
            right = left + int(rect['width'])
            cv2.rectangle(frame, (left,top),(right,btm),(255,255,255),3)

            emotion_result = ''
            dy = top + 0
            for e in emotion:
                emotion_result = e + ':' + str(emotion[e])
                dy = dy + 15
                cv2.putText(frame,emotion_result,(right + 10,dy), 0, 0.5,(255,255,255),1,cv2.LINE_AA)
            
            positive_valence = float(emotion['happiness'])
            negative_valence = float(emotion['sadness'])
            valence = positive_valence - negative_valence
            cv2.putText(frame,'Valence:'+ str(valence),(right + 10,dy+15), 0, 0.5,(255,255,255),1,cv2.LINE_AA)

        writer.write(frame)
        filepath = os.path.join('output', 'frame_{:04d}.jpg'.format(count))
        cv2.imwrite(filepath, frame)
        count += 1
        cv2.imshow('Raw Frame', frame)
        
        print("")

        k = cv2.waitKey(1)
        if k == 27: # wait for ESC key to exit
            print()
            break
# except Exception as e:
except IOError as e:
    print("[Errno {0}] {1}".format(e.errno, e.strerror))