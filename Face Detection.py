import cv2  # OpenCV kütüphanesini içe aktarır.
import mediapipe as mp  # MediaPipe kütüphanesini içe aktarır.

# Video dosyasını açar.
cap = cv2.VideoCapture("video3.mp4")

# Yüz algılama modelini yükler.
mpFaceDetection = mp.solutions.face_detection #yüz tespiti çözümü alınır. Bu, mediapipe kütüphanesinin yüz tespiti özelliğini kullanabilmek için gerekli olan sınıf ve metotları içerir.
faceDetection = mpFaceDetection.FaceDetection(0.20) # Hassaslığı 0 ile 1 arasındadır. Hassaslığı arttırdıkça yüzleri tespit etmesi zorlaşır. Çok fazla azaltırsak da her yeri yüz olarak algılamaya çalışır.

# Çizim işlevlerini kullanabilmek için gerekli olan araçları yükler.
mpDraw = mp.solutions.drawing_utils

while True:
    # Video akışından bir kare alır.
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV görüntü biçimini MediaPipe biçimine dönüştürür.

    # Yüz tespiti işlemini gerçekleştirir.
    results = faceDetection.process(imgRGB)

    # Tespit sonuçlarını kontrol eder.
    if results.detections:
        # Her bir tespit için döngü yapar.
        for id, detection in enumerate(results.detections):
            bboxC = detection.location_data.relative_bounding_box  # Bounding box koordinatlarını alır.
            h, w, _ = img.shape  # Görüntü boyutlarını alır.
            bbox = int(bboxC.xmin*w), int(bboxC.ymin*h), int(bboxC.width*w), int(bboxC.height*h)  # Bounding box koordinatlarını orijinal görüntü boyutuna dönüştürür.
            cv2.rectangle(img, bbox, (0,255,255), 2)  # Bounding box'u çizer.

    # Sonuçları ekranda gösterir.
    cv2.imshow("img", img)
    cv2.waitKey(10)  # Klavyeden bir tuşa basılmasını bekler.

# Video işleme işlemi tamamlandıktan sonra, pencereleri kapatır.
cap.release()
cv2.destroyAllWindows()

    
    
    
    
    
    
    