from os import listdir
from os.path import isfile, join
import xml.etree.ElementTree as ET
import pandas as pd

'''
This is a pre-processing script will load csv file containing the pose data corresponding to camera image frames.
It will then read the xml and csv files and add the pose data from the csv file, to the xml files.
Make sure the naming format of the xmls are "<number>.xml"
'''

# SETTINGS
path = '../airplane_annotation/'
csv_filename = '/home/ubuntu/catkin_ws/src/Northstar/airplane_loader/scripts/image_pose_data.csv'
print("\nCurrent annotation path : " + path)
print("\ncsv path with filename : " + csv_filename)

# Load files in the directory
onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
files = []

# Load the csv file data. We will be grabbing the rows from this later.
csv_data = pd.read_csv(csv_filename)

# Put all valid filenames in files list
for f in onlyfiles:
    if '~' not in f:
        files.append(f)

for filename in files:
    #filename = path + filename

    tree = ET.parse(path + filename)
    root = tree.getroot()

    for obj in root.iter('object'):           # loop through all the detected objects
        if obj[0].text == 'CargoDoor':        # check if the detected object is a 'CargoDoor'
            x = ET.SubElement(obj, 'pose_x')  # Add pose sub-elements
            y = ET.SubElement(obj, 'pose_y')
            z = ET.SubElement(obj, 'pose_z')

    # Grab the row corresponding to the filename. No Donald.
    try:
        current_row = csv_data.iloc[int(filename[:-4])]
    except ValueError:
        print("\nExpected xml filename mismatch. Expecting a '<number>.xml', however got " + filename)

    pose_x_from_csv = current_row.pose_x    # Assign pose data from the csv file
    pose_y_from_csv = current_row.pose_y * 0
    pose_z_from_csv = current_row.pose_z * 0
    
    # pose_x_from_csv = np.random.uniform(0, 10)    # Assign pose data from the csv file
    # pose_y_from_csv = np.random.uniform(2, 15)
    # pose_z_from_csv = np.random.uniform(0, 0.5)

    for obj in root.iter('object'):
        if obj[0].text == 'CargoDoor':
            obj[5].text = str(pose_x_from_csv)  # pose_x
            obj[6].text = str(pose_y_from_csv)  # pose_y
            obj[7].text = str(pose_z_from_csv)  # pose_z

    tree.write(path + filename)

print("Preprocessing done.")
