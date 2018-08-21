import cv2
from os import listdir
from os.path import isfile, join

''' Takes in a folder full of numbered image files and sorts them, and writes it into a video file'''

path = '/home/ubuntu/tensorflow_data/modified_YOLO_image_data/'
onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
files = []

for f in onlyfiles:
    if '~' not in f:
        files.append(f)

files.sort(key=lambda f: int(filter(str.isdigit,f)))

fourcc = cv2.cv.CV_FOURCC(*'XVID')
video = cv2.VideoWriter('outasdpy.avi', fourcc, 4, (640,480))  # filename, ~, frame_rate, size)

for filename in files:
    image = cv2.imread(path + filename)
    video.write(image)
    print("attaching file " + str(filename))

cv2.destroyAllWindows()
video.release()

print("Done")
