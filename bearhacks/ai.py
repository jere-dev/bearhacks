import cv2

#import training
#haarcascade_frontalface_default
trained_face_data = cv2.CascadeClassifier('cascade/cascade.xml')

# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#  print("Cannot open camera")
#  exit()

# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()

#     # if frame is read correctly ret is True
#     if not ret:
#         print("Can't receive frame (stream end?). Exiting ...")
#         break

#     grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     #get cordinates for face
#     face_cordinates = trained_face_data.detectMultiScale(grayscale_img)

#     if len(face_cordinates):
#         for fc in face_cordinates:
#             (x, y, w, h) = fc
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 10)

#     cv2.imshow("idkwtfid", frame)

#     if cv2.waitKey(1) == ord('q'):
#         break

# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()
 


for i in range(1, 74):
    #read img and grayscale
    img = cv2.imread('middle-finger-dataset/mid (' + str(i) + ').jpg')
    print(i)
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #get cordinates for face
    face_cordinates = trained_face_data.detectMultiScale(grayscale_img)
    (x, y, w, h) = face_cordinates[0]

    roi = img[y:y+h, x:x+w] 
    # applying a gaussian blur over this new rectangle area 
    roi = cv2.GaussianBlur(roi, (23, 23), 30) 
    # impose this blurred image on original image to get final image 
    img[y:y+roi.shape[0], x:x+roi.shape[1]] = roi 

    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 10)

    cv2.imshow("idkwtfid", img)
    cv2.waitKey()

#C:/apps/opencv/build/x64/vc15/bin/opencv_traincascade.exe -data cascade/ -vec pos.vec -bg neg.txt -w 24 -h 24 -numPos 70 -numNeg 150 -numStages 10
#C:/apps/opencv/build/x64/vc15/bin/opencv_createsamples.exe -info pos.txt -w 24 -h 24 -num 1000 -vec pos.vec