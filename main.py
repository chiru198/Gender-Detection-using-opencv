import cv2

def detectFace(net, frame, confidence_threshold=0.7):
    """
    Detect faces in a given frame using the provided neural network.

    Args:
    - net: The neural network for face detection.
    - frame: The image frame in which to detect faces.
    - confidence_threshold: Minimum confidence to consider a detection.

    Returns:
    - resultImg: The image frame with rectangles drawn around detected faces.
    - faceBoxes: List of coordinates for the detected faces.
    """
    frameOpencvDNN = frame.copy()
    frameHeight = frameOpencvDNN.shape[0]
    frameWidth = frameOpencvDNN.shape[1]

    # Create a blob from the image and perform forward pass to get detections
    blob = cv2.dnn.blobFromImage(frameOpencvDNN, 1.0, (227, 227), [124.96, 115.97, 106.13], swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()

    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDNN, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    
    return frameOpencvDNN, faceBoxes

# Define model paths
faceProto = r'E:\New folder\opencv_face_detector(1).pbtxt'
faceModel = r'E:\New folder\opencv_face_detector_uint8(1).pb'
genderProto = r'E:\New folder\gender_deploy(1).prototxt'
genderModel = r'E:\New folder\gender_net(1).caffemodel'

# Define gender list
genderList = ['Male', 'Female']

# Load models
faceNet = cv2.dnn.readNet(faceModel, faceProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Open video capture
video = cv2.VideoCapture(0)  # Change to the path of your video file if needed
if not video.isOpened():
    print("Error: Video capture device could not be opened.")
    exit()

# Initialize counters for male and female
male_count = 0
female_count = 0

padding = 20

while True:
    hasFrame, frame = video.read()
    if not hasFrame:
        print("End of video file reached or no frames to read.")
        break

    # Detect faces
    resultImg, faceBoxes = detectFace(faceNet, frame)

    for faceBox in faceBoxes:
        face = frame[max(0, faceBox[1] - padding):min(faceBox[3] + padding, frame.shape[0] - 1),
                     max(0, faceBox[0] - padding):min(faceBox[2] + padding, frame.shape[1] - 1)]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), [124.96, 115.97, 106.13], swapRB=True, crop=False)
        
        # Predict gender
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]

        # Update counters
        if gender == 'Male':
            male_count += 1
        else:
            female_count += 1
            
        # Display gender on the image
        cv2.putText(resultImg, f'Gender: {gender}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    # Show result image
    cv2.imshow("Detecting Gender", resultImg)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

# Print the final counts
print(f"Total males detected: {male_count}")
print(f"Total females detected: {female_count}")

# Release video capture and close windows
video.release()
cv2.destroyAllWindows()
