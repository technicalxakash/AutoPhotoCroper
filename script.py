# import cv2
# import os
# import numpy as np

# # ===== CONFIG =====
# input_folder = "input_images"
# output_folder = "output_passport"
# os.makedirs(output_folder, exist_ok=True)

# # Passport size (pixels) — change if needed
# PASSPORT_WIDTH = 413   # ~35mm at 300dpi
# PASSPORT_HEIGHT = 531  # ~45mm at 300dpi

# # Load face detector
# face_cascade = cv2.CascadeClassifier(
#     cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# )


# def crop_passport(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     faces = face_cascade.detectMultiScale(
#         gray,
#         scaleFactor=1.2,
#         minNeighbors=5,
#         minSize=(100, 100)
#     )

#     if len(faces) == 0:
#         print("No face detected")
#         return None

#     # Take largest detected face
#     (x, y, w, h) = max(faces, key=lambda b: b[2] * b[3])

#     # Head center
#     cx = x + w // 2
#     cy = y + h // 2

#     # Passport crop region around face
#     crop_w = int(w * 2.2)
#     crop_h = int(h * 2.8)

#     start_x = max(cx - crop_w // 2, 0)
#     start_y = max(cy - int(h * 1.2), 0)

#     end_x = min(start_x + crop_w, image.shape[1])
#     end_y = min(start_y + crop_h, image.shape[0])

#     cropped = image[start_y:end_y, start_x:end_x]

#     # Resize to passport size
#     resized = cv2.resize(cropped, (PASSPORT_WIDTH, PASSPORT_HEIGHT))

#     return resized


# # ===== Batch Processing =====
# for file in os.listdir(input_folder):

#     path = os.path.join(input_folder, file)

#     img = cv2.imread(path)

#     if img is None:
#         continue

#     result = crop_passport(img)

#     if result is not None:
#         save_path = os.path.join(output_folder, file)
#         cv2.imwrite(save_path, result)
#         print("Saved:", save_path)

# print("Batch Processing Done ✅")




# import cv2
# import mediapipe as mp
# import numpy as np
# import os
# import math

# input_folder = "input_images"
# output_folder = "output_passport"
# os.makedirs(output_folder, exist_ok=True)

# PASSPORT_SIZE = (413, 531)

# mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)


# def align_and_crop(image):

#     rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = mp_face.process(rgb)

#     if not results.multi_face_landmarks:
#         return None

#     landmarks = results.multi_face_landmarks[0].landmark
#     h, w = image.shape[:2]

#     # Left eye & right eye points
#     left_eye = landmarks[33]
#     right_eye = landmarks[263]

#     x1, y1 = int(left_eye.x * w), int(left_eye.y * h)
#     x2, y2 = int(right_eye.x * w), int(right_eye.y * h)

#     # ---- Calculate angle ----
#     angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

#     # ---- Rotate image ----
#     center = (w // 2, h // 2)
#     M = cv2.getRotationMatrix2D(center, angle, 1)
#     rotated = cv2.warpAffine(image, M, (w, h))

#     # ---- Face bounding box again after rotation ----
#     rgb2 = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
#     results2 = mp_face.process(rgb2)

#     if not results2.multi_face_landmarks:
#         return None

#     lm = results2.multi_face_landmarks[0].landmark

#     xs = [int(p.x * w) for p in lm]
#     ys = [int(p.y * h) for p in lm]

#     x_min, x_max = min(xs), max(xs)
#     y_min, y_max = min(ys), max(ys)

#     face_w = x_max - x_min
#     face_h = y_max - y_min

#     cx = (x_min + x_max) // 2
#     cy = (y_min + y_max) // 2

#     crop_w = int(face_w * 2.2)
#     crop_h = int(face_h * 2.8)

#     start_x = max(cx - crop_w // 2, 0)
#     start_y = max(cy - int(face_h * 1.2), 0)

#     end_x = min(start_x + crop_w, w)
#     end_y = min(start_y + crop_h, h)

#     crop = rotated[start_y:end_y, start_x:end_x]

#     final = cv2.resize(crop, PASSPORT_SIZE)

#     return final


# for file in os.listdir(input_folder):

#     path = os.path.join(input_folder, file)
#     img = cv2.imread(path)

#     if img is None:
#         continue

#     result = align_and_crop(img)

#     if result is not None:
#         save_path = os.path.join(output_folder, file)
#         cv2.imwrite(save_path, result)
#         print("Saved:", file)

# print("Done ✅")


import cv2
import mediapipe as mp
import numpy as np
import os
import math

# ==============================
# CONFIG
# ==============================

input_folder = "input_images"
output_folder = "output_passport"

os.makedirs(output_folder, exist_ok=True)

# Passport size (pixels)
PASSPORT_WIDTH = 413
PASSPORT_HEIGHT = 531

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)


# ==============================
# ALIGN + CROP FUNCTION
# ==============================

def align_and_crop(image):

    h, w = image.shape[:2]

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark

    # Eye coordinates
    left_eye = landmarks[33]
    right_eye = landmarks[263]

    x1, y1 = int(left_eye.x * w), int(left_eye.y * h)
    x2, y2 = int(right_eye.x * w), int(right_eye.y * h)

    # ==========================
    # CALCULATE ROTATION ANGLE
    # ==========================
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1)

    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

    # ==========================
    # DETECT FACE AGAIN AFTER ROTATION
    # ==========================

    rgb2 = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
    results2 = face_mesh.process(rgb2)

    if not results2.multi_face_landmarks:
        return None

    lm = results2.multi_face_landmarks[0].landmark

    xs = [int(p.x * w) for p in lm]
    ys = [int(p.y * h) for p in lm]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    face_w = x_max - x_min
    face_h = y_max - y_min

    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2

    # ==========================
    # PASSPORT CROP REGION
    # ==========================

    crop_w = int(face_w * 2.2)
    crop_h = int(face_h * 2.8)

    start_x = max(cx - crop_w // 2, 0)
    start_y = max(cy - int(face_h * 1.2), 0)

    end_x = min(start_x + crop_w, w)
    end_y = min(start_y + crop_h, h)

    cropped = rotated[start_y:end_y, start_x:end_x]

    if cropped.size == 0:
        return None

    # ==========================
    # RESIZE TO PASSPORT SIZE
    # ==========================

    final = cv2.resize(cropped, (PASSPORT_WIDTH, PASSPORT_HEIGHT))

    return final


# ==============================
# BATCH PROCESSING
# ==============================

for file in os.listdir(input_folder):

    if not file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    path = os.path.join(input_folder, file)

    img = cv2.imread(path)

    if img is None:
        print("Failed:", file)
        continue

    result = align_and_crop(img)

    if result is not None:

        save_path = os.path.join(output_folder, file)
        cv2.imwrite(save_path, result)

        print("Saved:", file)

    else:
        print("Face not detected:", file)


print("Done ✅")