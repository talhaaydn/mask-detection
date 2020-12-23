from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os


def detect_and_predict_mask(frame, faceClassifier, cnnModel):
    # Frame üzerinden yüzleri tespit edebilmek için belli bir yapı oluşturuldu.
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))

    # Belli ölçülere sahip yüzler tespit edildi.
    faceClassifier.setInput(blob)
    detections = faceClassifier.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # Ekranda yüz olup olmadığı tahmin ediliyor.
        confidence = detections[0, 0, i, 2]

        # Ekrandaki yüz tespiti tahmini 50% den büyükse maske tespiti yapılacak.
        if confidence > 0.5:
            # Tespit edilen yüzün boyutları belirlendi.
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # Ekrandaki yüzün kapladığı alanı alır.
            # Yüzün görüntüsüne BDR to RGB dönüşümü yapılır.
            # Yüzün boyunu 224x224 ölçüsüne sabitler.
            # Yüzü array'e çevirdikten sonra preprocess işlemi uygular.
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # Ekranda bir yüz varsa tahmin işlemi uygulanır.
    if len(faces) > 0:
        # Ekranda tespit edilen yüzler için maske takılı olup olmadığını tahmin eder.
        # Bunun için kayıt etmiş olduğumuz model kullanılır.
        faces = np.array(faces, dtype="float32")
        preds = cnnModel.predict(faces, batch_size=64)

    # Yapılan işlemleri ekranda göstermek için yüzlerin yerleri ve tahmin değerleri geri döndürür.
    return (locs, preds)


# Kameradaki görüntüden yüz algılayabilmek için modeller yüklendi.
prototxtPath = r"face_detector/deploy.prototxt"
weightsPath = r"face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceClassifier = cv2.dnn.readNet(prototxtPath, weightsPath)

# Maske tespit etmek için oluşturduğumuz model yüklendi.
cnnModel = load_model("mask_detector.model")

# Kameranın açılması sağlandı.
vs = VideoStream(src=0).start()

# Q tuşuna basana kadar kameranın açık kalması sağlandı.
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=768)

    # Ekrandaki yüzleri tespit eder.
    # Maske takıp takmadıklarını tahmın eder.
    (locs, preds) = detect_and_predict_mask(frame, faceClassifier, cnnModel)

    # Tespit edilen her yüz için ekranda gösterme işlemi yapılıyor.
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # Ekranda gösterilecek text belirlenir.
        # Ekranda gösterilecek yüzün etrafındaki kutunun rengi belirlenir.
        label = "Maske Var" if mask > withoutMask else "Maske Yok"
        color = (0, 255, 0) if label == "Maske Var" else (0, 0, 255)

        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    cv2.imshow("Mask Detection - by Talha", frame)

    key = cv2.waitKey(1) & 0xFF

    # Q tuşuna basıldığında program çalışmayı durdurur.
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
