from imutils import paths
import face_recognition
import pickle
import cv2
import os
import imutils
import time

# updated from practice.ipynb

def save_ur_images(person="zaid",designation="developer"):
    cap = cv2.VideoCapture(0)
    count = 0
    folder = f'images/{person}_{"_".join(designation.split())}'
    if not os.path.exists(folder):
        os.makedirs(folder)
    while True:
        ret, frame = cap.read()
        # add a text for saving image
        cv2.putText(frame, f'Press "s" to save image', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(frame, f'Press "q" to quit', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)
        if key == ord('s'):
            cv2.imwrite(f"{folder}/{person}_{count}.jpg", frame)
            count += 1
        if key == ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()

def load_face_encodings():
    return 'encodings/faces'

def save_face_encodings(st,model='hog',stlit=False):
    
    file_encoding = f'encodings/faces'
    if not os.path.exists('encodings'):
        os.makedirs('encodings')
    imagePaths = list(paths.list_images(f'images'))
    knownEncodings = []
    knownNames = []
    # loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        name = imagePath.split('\\')[1]
        if stlit:
            st.write(f'[INFO] processing image {i + 1}/{len(imagePaths)}: {name}')
        else:
            print(f'[INFO] processing image {i + 1}/{len(imagePaths)}: {name}')
        # print(imagePath,name)
        # load the input image and convert it from BGR (OpenCV ordering) to dlib ordering (RGB)
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #Use Face_recognition to locate faces
        boxes = face_recognition.face_locations(rgb,model=model)
        # compute the facial embedding for the face
        encodings = face_recognition.face_encodings(rgb, boxes)
        # loop over the encodings
        for encoding in encodings:
            knownEncodings.append(encoding)
            knownNames.append(name)
    #save emcodings along with their names in dictionary data
    data = {"encodings": knownEncodings, "names": knownNames}
    #use pickle to save data into a file for later use
    f = open(file_encoding, "wb")
    f.write(pickle.dumps(data))
    f.close()
    return file_encoding

def start_camera(encoding):
    # load the known faces and embeddings saved in last file
    data = pickle.loads(open(encoding, "rb").read())

    #find path of xml file containing haarcascade file 
    cascPathface = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
    # load the harcaascade in the cascade classifier
    faceCascade = cv2.CascadeClassifier(cascPathface)

    video_capture = cv2.VideoCapture(0)
    # loop over frames from the video file stream
    print("Streaming started")
    while True:
        # grab the frame from the threaded video stream
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray,
                                            scaleFactor=1.1,
                                            minNeighbors=5,
                                            minSize=(60, 60),
                                            flags=cv2.CASCADE_SCALE_IMAGE)
    
        # convert the input frame from BGR to RGB 
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # the facial embeddings for face in input
        encodings = face_recognition.face_encodings(rgb)
        names = []
        # loop over the facial embeddings incase
        # we have multiple embeddings for multiple fcaes
        for encoding in encodings:
        #Compare encodings with encodings in data["encodings"]
        #Matches contain array with boolean values and True for the embeddings it matches closely
        #and False for rest
            matches = face_recognition.compare_faces(data["encodings"],
            encoding)
            #set name =inknown if no encoding matches
            name = "Unknown_unknown"
            # check to see if we have found a match
            if True in matches:
                #Find positions at which we get True and store them
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    #Check the names at respective indexes we stored in matchedIdxs
                    name = data["names"][i]
                    #increase count for the name we got
                    counts[name] = counts.get(name, 0) + 1
                #set name which has highest count
                name = max(counts, key=counts.get)
    
    
            # update the list of names
            names.append(name)
            # loop over the recognized faces
            for ((x, y, w, h), name) in zip(faces, names):
                # rescale the face coordinates
                # draw the predicted face name on the image
                try:
                    n,d = name.split('_')
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    cv2.putText(frame, f'Person: {n.upper()}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 255), 2)
                    cv2.putText(frame, f'Designation: {d.lower()}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX,.5, (0, 255, 255), 1)
                except:
                    pass
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()

def delete_dir(dir):
    import shutil
    dir = f'images/{dir}'
    if os.path.exists(dir):
        try:
            shutil.rmtree(dir)
            return True
        except:
            return False
    else:
        return False
