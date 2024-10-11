import numpy as np
import cv2
import cvface
import face_recognition
from flask import Flask, request, jsonify

app = Flask(__name__)
@app.route('/recognize', methods=['POST'])
def recognize():
    unknown_image = request.files['image']
    known_image = request.files["known_image"]
    tolerance = float(request.form.get('tolerance', 0.6))
    if not unknown_image:
        return "No image uploaded", 400
    if not known_image:
        return "No unknown image uploaded", 400
    
    # Read the image file using NumPy
    img = cv2.imdecode(np.fromstring(unknown_image.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    match = cvface.match(known_image, unknown_image, tolerance)

    result = {
        #tidak akurat jika menggunakan opencv
        #'smile': cvface.smile_detection(img),
        'smile': cvface.smile_detection_fr(unknown_image),
        'face_detected': cvface.face_detection(img),
        'mask': cvface.mask_detection(img),
        'match': match
    }

    print(result)
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)