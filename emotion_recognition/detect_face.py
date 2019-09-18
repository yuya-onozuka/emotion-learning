import requests
import json

subscription_key = '760f908d0ed34f7188ee667ce9a316a0'
assert subscription_key

face_api_url = 'https://emotion-recognition.cognitiveservices.azure.com/face/v1.0/detect'

image_url = 'https://upload.wikimedia.org/wikipedia/commons/3/37/Dagestani_man_and_woman.jpg'

headers = {'Ocp-Apim-Subscription-Key': subscription_key}

params = {
    'returnFaceId': 'false',
    'returnFaceLandmarks': 'false',
    # 'returnFaceAttributes': 'age,gender,headPose,smile,facialHair,glasses,emotion,hair,makeup,occlusion,accessories,blur,exposure,noise',
    'returnFaceAttributes': 'emotion',
}

response = requests.post(face_api_url, params=params,
                         headers=headers, json={"url": image_url})
print(json.dumps(response.json()))
print(response.json())