# import cv2
# import mediapipe as mp
# import numpy as np
# import os
# import math

# # ==============================
# # CONFIG
# # ==============================

# input_folder = "input_images"
# output_folder = "output_passport"

# os.makedirs(output_folder, exist_ok=True)

# # Passport size (pixels)
# PASSPORT_WIDTH = 413
# PASSPORT_HEIGHT = 531

# # MediaPipe Face Mesh
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)


# # ==============================
# # ALIGN + CROP FUNCTION
# # ==============================

# def align_and_crop(image):

#     h, w = image.shape[:2]

#     rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = face_mesh.process(rgb)

#     if not results.multi_face_landmarks:
#         return None

#     landmarks = results.multi_face_landmarks[0].landmark

#     # Eyes
#     left_eye = landmarks[33]
#     right_eye = landmarks[263]

#     x1, y1 = int(left_eye.x * w), int(left_eye.y * h)
#     x2, y2 = int(right_eye.x * w), int(right_eye.y * h)

#     angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

#     center = (w // 2, h // 2)
#     M = cv2.getRotationMatrix2D(center, angle, 1)

#     rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

#     # Detect again
#     rgb2 = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
#     results2 = face_mesh.process(rgb2)

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

#     # Passport ratio
#     passport_ratio = PASSPORT_WIDTH / PASSPORT_HEIGHT

#     # Desired crop height based on face size
#     crop_h = int(face_h * 2.5)
#     crop_w = int(crop_h * passport_ratio)

#     start_x = max(cx - crop_w // 2, 0)
#     start_y = max(y_min - int(face_h * 0.6), 0)

#     end_x = min(start_x + crop_w, w)
#     end_y = min(start_y + crop_h, h)

#     cropped = rotated[start_y:end_y, start_x:end_x]

#     if cropped.size == 0:
#         return None

#     final = resize_with_padding(cropped, PASSPORT_WIDTH, PASSPORT_HEIGHT)

#     return final

# # ==============================
# # BATCH PROCESSING
# # ==============================

# for file in os.listdir(input_folder):

#     if not file.lower().endswith((".jpg", ".jpeg", ".png")):
#         continue

#     path = os.path.join(input_folder, file)

#     img = cv2.imread(path)

#     if img is None:
#         print("Failed:", file)
#         continue

#     result = align_and_crop(img)

#     if result is not None:

#         save_path = os.path.join(output_folder, file)
#         cv2.imwrite(save_path, result)

#         print("Saved:", file)

#     else:
#         print("Face not detected:", file)


# print("Done ✅")







import cv2
import mediapipe as mp
import numpy as np
import os
import math

# ==========================================
# CONFIG
# ==========================================

input_folder = "input_images"
output_folder = "output_passport"

os.makedirs(output_folder, exist_ok=True)

# Passport size (35mm x 45mm approx at 300dpi)
PASSPORT_WIDTH = 413
PASSPORT_HEIGHT = 531

# ==========================================
# MEDIAPIPE SETUP (Use 0.10.14 version)
# ==========================================

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)


# ==========================================
# RESIZE WITH PADDING (NO DISTORTION)
# ==========================================

def resize_with_padding(img, target_w, target_h):

    h, w = img.shape[:2]

    scale = min(target_w / w, target_h / h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # White background
    canvas = np.full((target_h, target_w, 3), (255, 255, 255), dtype=np.uint8)

    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return canvas


# ==========================================
# ALIGN + CROP FUNCTION
# ==========================================

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

    # Calculate rotation angle
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1)

    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

    # Detect again after rotation
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

    # Passport aspect ratio
    passport_ratio = PASSPORT_WIDTH / PASSPORT_HEIGHT

    # Crop height relative to face
    crop_h = int(face_h * 2.5)
    crop_w = int(crop_h * passport_ratio)

    start_x = max(cx - crop_w // 2, 0)
    start_y = max(y_min - int(face_h * 0.6), 0)

    end_x = min(start_x + crop_w, w)
    end_y = min(start_y + crop_h, h)

    cropped = rotated[start_y:end_y, start_x:end_x]

    if cropped.size == 0:
        return None

    # Resize without distortion
    final = resize_with_padding(cropped, PASSPORT_WIDTH, PASSPORT_HEIGHT)

    return final


# ==========================================
# BATCH PROCESSING
# ==========================================

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