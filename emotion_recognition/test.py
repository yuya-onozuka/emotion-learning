import cognitive_face as CF

KEY = '760f908d0ed34f7188ee667ce9a316a0'  # Replace with a valid Subscription Key here.
CF.Key.set(KEY)

BASE_URL = 'https://emotion-recognition.cognitiveservices.azure.com/face/v1.0'  # Replace with your regional Base URL
CF.BaseUrl.set(BASE_URL)

img_url = 'https://raw.githubusercontent.com/Microsoft/Cognitive-Face-Windows/master/Data/detection1.jpg'
result = CF.face.detect(img_url)
print result