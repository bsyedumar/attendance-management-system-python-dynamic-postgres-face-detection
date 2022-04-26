import psycopg2
from tkinter import *
from PIL import ImageTk, Image
import cv2
import os
import face_recognition
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Import the 'config' funtion from the config.py file
from config import config

# Establish a connection to the database by creating a cursor object

# Obtain the configuration parameters
params = config()
# Connect to the PostgreSQL database
conn = psycopg2.connect(**params)
cur = conn.cursor()
# Create a new cursor
cur.execute("SELECT facial_img FROM att_entry")
myresult = cur.fetchall()
# values = myresult[0]
values = [myresult[0] for myresult in cur]
values = bytes(values)
# print(type(values))
with open("db_img_read.jpg", "wb") as file:
    img = file.write(values)
root = Tk()
path = r"/Users/umar/Downloads/6thSem/AI_Lab/images/db_img_read.jpg"
#img = ImageTk.PhotoImage(Image.open(path))
panel = Label(root, image=img)
panel.pack(side="bottom", fill="both", expand="yes")
root.mainloop()


def markAttendance(name):
    with open("attendance_list.csv", "r+") as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(",")
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime("%H:%M:%S")
            f.writelines(f"\n{name},{dtString}")


prefix = "images/"
# finds number of items in folder dataset
image_type = ".jpg"
img = input("Enter image 1 name: ")
# img_test = input("Enter test image name: ")
image_path = "/Users/umar/Downloads/6thSem/AI_Lab/images/db_img_read.png"
# image_path = prefix + "umar" + image_type
image_path_test = prefix + img_test + image_type
# image_path1 = prefix + "img" + image_type

imgUmar = face_recognition.load_image_file(image_path)
imgUmar = cv2.cvtColor(imgUmar, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file(image_path_test)
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)
# using cnn kills python progrsm
faceLoc = face_recognition.face_locations(
    imgUmar, number_of_times_to_upsample=2, model="hog"
)[0]
encodeUmar = face_recognition.face_encodings(imgUmar, num_jitters=5, model="hog")[0]
cv2.rectangle(
    imgUmar, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2
)

faceLocTest = face_recognition.face_locations(
    imgTest, number_of_times_to_upsample=2, model="hog"
)[0]
encodeTest = face_recognition.face_encodings(imgTest, num_jitters=5, model="cnn")[0]
cv2.rectangle(
    imgTest,
    (faceLocTest[3], faceLocTest[0]),
    (faceLocTest[1], faceLocTest[2]),
    (255, 0, 255),
    2,
)

results = face_recognition.compare_faces([encodeUmar], encodeTest, 0.6)
faceDis = face_recognition.face_distance([encodeUmar], encodeTest)
print(results, faceDis)
cv2.putText(
    imgUmar,
    f"{results} {round(faceDis[0],2)}",
    (50, 50),
    cv2.FONT_HERSHEY_COMPLEX,
    1,
    (0, 0, 255),
    2,
)
cv2.putText(
    imgTest,
    f"{results} {round(faceDis[0],2)}",
    (50, 50),
    cv2.FONT_HERSHEY_COMPLEX,
    1,
    (0, 0, 255),
    2,
)

face_match_percentage = (1 - faceDis) * 100
for i, face_distance in enumerate(faceDis):
    print(
        "The test image has a distance of {:.2} from known image {}".format(
            face_distance, i
        )
    )
    print("- comparing with a tolerance of 0.6 {}".format(face_distance < 0.6))
    print(
        "Face Match Percentage = ", np.round(face_match_percentage, 4)
    )  # upto 4 decimal places
if results == [True] and faceDis < 0.35:
    # Createname = "SELECT first_name FROM att_entry"
    cur.execute("SELECT first_name FROM att_entry")
    myresult1 = cur.fetchone()
    name = myresult1[0]
    print(name)
    markAttendance(name)
cv2.imshow("Umar ", imgUmar)
plt.figure()
cv2.imshow("Umar Test ", imgTest)
cv2.waitKey(0)
cur.close()
conn.close()
cv2.destroyAllWindows()


# b = base64.b64decode(values)
#     db_img = Image.open(io.BytesIO(b))
#     db_img.save(b, "PNG")
#     b.seek(0)
#     db_img = b.read()
#     dataBytesIO = io.BytesIO(db_img)
#     Image.open(dataBytesIO)

