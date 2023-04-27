import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'images_attendance'
images = []         # the list of all the images
names = []          # the file names of the images
img_list = os.listdir(path)         # grabbing the images in the folder
#print(img_list)

# loading the images
for img in img_list:
    current_img = cv2.imread(f'{path}/{img}')
    images.append(current_img)
    names.append(os.path.splitext(img)[0])      # removing the .jpg from file name
#print(names)


# creating a function to calculate the encoding of each image
def find_encodings(images):
    encodings_list = []
    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)      # from BGR to RGB
        encoding = face_recognition.face_encodings(image)[0]
        encodings_list.append(encoding)
    return encodings_list


known_list_encodings = find_encodings(images)          # finding the encodings
# print(len(known_list_encodings))         # to check
print('Encoding complete')          # to check


# defining a function for the attendance

def attendance(name):
    with open('attendance.csv', 'r+') as f:
        my_data_list = f.readlines()
        name_list = []
        for line in my_data_list:
            entry = line.split(',')
            name_list.append(entry[0])
        if name not in name_list:
            this_moment = datetime.now()
            time_string = this_moment.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{time_string}')


def description(name):


    if name == 'JAMIE':
        cv2.putText(cap, 'A proud lannister', (x1 + 6, y2 + 52),
                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, .8, (0, 0, 0), 1)

    if name == 'JON':
        cv2.putText(cap, 'You know nothing!', (x1 + 6, y2 + 52),
                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, .8, (0, 0, 0), 1)


    if name == 'UNKNOWN':
        cv2.putText(cap, "Who is this?", (x1 + 6, y2 + 52),
                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, .8, (0, 0, 0), 1)



# initializing the webcam to find the real time images and their encodings
vid = cv2.VideoCapture(0)

while True:
    success, cap = vid.read()
    img_small = cv2.resize(cap, (0, 0), None, 0.25, 0.25)
    img_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

    # finding the encoding of the webcam images
    current_face_frame_location = face_recognition.face_locations(img_small)
    encoding_of_current_frame = face_recognition.face_encodings(img_small, current_face_frame_location)

    # finding the matches between encodings
    # we will iterate through all the faces in our current frame
    # then we will compare them with saved known encodings

    for new_face_encode, new_face_loc in zip(encoding_of_current_frame, current_face_frame_location):
        compared_encodings = face_recognition.compare_faces(known_list_encodings, new_face_encode)
        distances = face_recognition.face_distance(known_list_encodings, new_face_encode)
        #print(distances)

        # now the one with the lowest distance value will be the accurate face
        matching_index = np.argmin(distances)

        if compared_encodings[matching_index]:
            name = names[matching_index].upper()
            #print(name)

            # printing a rectangle around the face
            y1, x2, y2, x1 = new_face_loc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(cap, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(cap, (x1-1, y2), (x2+1, y2+55), (255, 0, 255), cv2.FILLED)
            cv2.putText(cap, name, (x1+6, y2+27), cv2.FONT_HERSHEY_TRIPLEX, 1,
                        (255, 255, 255), 2)

            description(name)
            attendance(name)

        else:
            name = 'UNKNOWN'
            y1, x2, y2, x1 = new_face_loc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(cap, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(cap, (x1 - 1, y2), (x2 + 1, y2 + 55), (255, 0, 255), cv2.FILLED)
            cv2.putText(cap, name, (x1 + 6, y2 + 27), cv2.FONT_HERSHEY_TRIPLEX, 1,
                        (255, 255, 255), 2)
            description(name)


    cv2.imshow("Camera", cap)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
