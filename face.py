import face_recognition
import math

#fungsi fungsi di class ini menggunakan library face_recognition

#buat fungsi deteksi masker di wajah
def mask_detection(image):
	face_landmarks_list = face_recognition.face_landmarks(image)
	# Dapatkan titik-titik mata
	left_eye = face_landmarks_list[0]['left_eye'][0]
	right_eye = face_landmarks_list[0]['right_eye'][0]

	# Hitung jarak antara mata
	distance = ((left_eye[0] - right_eye[0]) ** 2 + (left_eye[1] - right_eye[1]) ** 2) ** 0.5

	# Bandingkan nilai-nilai dengan nilai-nilai rujukan untuk masker
	is_masked = distance > 20
	return is_masked

#buat fungsi deteksi senyum di wajah
def smile_detection(image):
	face_landmarks_list = face_recognition.face_landmarks(image)
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


# fungsi bolean apakah wajah terdeksi atau tidak 
def recognize(image):
	face_landmarks_list = face_recognition.face_landmarks(image)
	if len(face_landmarks_list) == 0:
		return False
	else:
		return True
	
#bandingkan dua gambar, apakah wajah memiliki kemiripan 70% atau lebih
def match(known_image, unknown_image, tolerance=0.7):
	known_face_encoding = face_recognition.face_encodings(known_image)[0]
	unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]
	return face_recognition.compare_faces([known_face_encoding], unknown_face_encoding, tolerance)
