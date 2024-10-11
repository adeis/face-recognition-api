import cv2
import face_recognition
import math
from io import BytesIO

# fungsi fungsi di class ini menggunakan library cv2
# Load Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

def face_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return len(faces) > 0

def smile_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    smiles = smile_cascade.detectMultiScale(gray, 1.8, 20)
    return len(smiles) > 0

def mask_detection(image):
    # Simple mask detection based on color (assumes wearing blue masks)
    lower_blue = (100, 150, 0)
    upper_blue = (140, 255, 255)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    return cv2.countNonZero(mask) > 1000

#bandingkan dua gambar, apakah wajah memiliki kemiripan 60% atau lebih dengan opencv
def match(known_image, unknown_image, tolerance=0.6):
    try:
        # Terima gambar wajah yang akan diidentifikasi
        unknown_image = face_recognition.load_image_file(unknown_image)
        # Ekstrak fitur wajah dari gambar yang diidentifikasi
        unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]
        #print(unknown_face_encoding)

        known_image = face_recognition.load_image_file(known_image)
        # Ekstrak fitur wajah dari gambar yang diidentifikasi
        known_face_encoding = face_recognition.face_encodings(known_image)[0]

        # Bandingkan fitur wajah yang diidentifikasi dengan fitur wajah referensi
        match = face_recognition.compare_faces([known_face_encoding], unknown_face_encoding, tolerance)

        # Jika wajah yang diidentifikasi cocok dengan wajah referensi, maka return True
        if match[0]:
            return True
        else:
            return False
       
    except Exception as e:
        print(e)
        return False
def smile_detection_fr(image):
    try:
        fr_image = face_recognition.load_image_file(image)
        face_landmarks_list = face_recognition.face_landmarks(fr_image)
        # Dapatkan titik-titik mulut
        mouth_left = face_landmarks_list[0]['top_lip'][0]
        mouth_right = face_landmarks_list[0]['top_lip'][-1]
        nose_tip = face_landmarks_list[0]['nose_tip'][0]

        # Hitung jarak antara sudut mulut dan hidung
        distance = ((mouth_left[0] - nose_tip[0]) ** 2 + (mouth_left[1] - nose_tip[1]) ** 2) ** 0.5

        # Hitung sudut antara garis yang menghubungkan sudut mulut dan hidung
        angle = math.degrees(math.atan2(mouth_right[1] - mouth_left[1], mouth_right[0] - mouth_left[0]))

        # Bandingkan nilai-nilai dengan nilai-nilai rujukan untuk senyum
        is_smiling = distance > 50 and angle > 10
        return is_smiling
    except:
        return False
    

def recognize(known_image_file,unknown_image_file,tolerance=0.6):
    #if 'known_image' not in request.files or 'unknown_image' not in request.files:
    #    return jsonify({'error': 'Missing image parameters'}), 400

    #known_image_file = request.files['known_image']
    #unknown_image_file = request.files['unknown_image']
    #tolerance = float(request.form.get('tolerance', 0.6))  # Default tolerance is 0.6

    try:
        # Convert images to compatible format and load encodings
        known_image = face_recognition.load_image_file(BytesIO(known_image_file.read()))
        unknown_image = face_recognition.load_image_file(BytesIO(unknown_image_file.read()))

        # Reset the stream for next read (if necessary)
        known_image_file.stream.seek(0)
        unknown_image_file.stream.seek(0)

        known_face_encodings = face_recognition.face_encodings(known_image)
        unknown_face_encodings = face_recognition.face_encodings(unknown_image)

        if not known_face_encodings or not unknown_face_encodings:
            return "No faces detected in one or both images"
            #jsonify({'error': 'No faces detected in one or both images'}), 400

        # Compare faces
        results = face_recognition.compare_faces([known_face_encodings[0]], unknown_face_encodings[0], tolerance)
        if results[0]:
            return "Wajah Dikenali"
        else:
            return "Wajah Tidak dikenali"

    except Exception as e:
        return str(e)