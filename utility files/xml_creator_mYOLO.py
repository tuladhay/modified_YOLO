from os import listdir
from os.path import isfile, join
import xml.etree.ElementTree as ET
import pandas as pd

'''
This script will take the images and the csv data generated from the modified YOLO express, which is the dataset
generated by using YOLO itself for labeling the bounding boxes, and then using the ground truth data for pose.
This way, we don't need a human to label the data, and thus can generate much more training data to mYOLO to learn the
pose.

This is not going to print the XML in pretty print format. There will be no whitespace.
'''

# SETTINGS
path = '/home/ubuntu/tensorflow_data/DATA/data1_images/'
csv_filename = '/home/ubuntu/tensorflow_data/CSV_backup/image_pose_data_express_data1.csv'
xml_write_path = '/home/ubuntu/tensorflow_data/data1_renamed_img_anno/data1_annotation/'
print("\nCurrent image path : " + path)
print("\ncsv path with filename : " + csv_filename)
print("\nxml write path : " + xml_write_path)

OFFSET = 702

# Load files in the directory
onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
files = []

# Load the csv file data. We will be grabbing the rows from this later.
csv_data = pd.read_csv(csv_filename)

# Put all valid filenames in files list
for f in onlyfiles:
    if '~' not in f:
        files.append(f)

for filename in files:    # keep in mind the filenames are not sorted
    # read the csv file row corresponding the image number and extract the pose

    try:
        current_row = csv_data.iloc[int(filename[:-5])]
    except ValueError:
        print("\nExpected xml filename mismatch. Expecting a '<number>.xml', however got " + filename)
    pose_x_from_csv = current_row.pose_x  # Assign pose data from the csv file
    pose_y_from_csv = current_row.pose_y * 0
    pose_z_from_csv = current_row.pose_z * 0

    xmin_from_csv = current_row.xmin
    ymin_from_csv = current_row.ymin
    xmax_from_csv = current_row.xmax
    ymax_from_csv = current_row.ymax

    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = "data1_images"
    ET.SubElement(annotation, "filename").text = str(int(filename[:-5]) + OFFSET) + ".jpeg"
    ET.SubElement(annotation, "path").text = path + str(int(filename[:-5]) + OFFSET) + ".jpeg"

    size_element = ET.SubElement(annotation, "size")
    ET.SubElement(size_element, "width").text = "640"
    ET.SubElement(size_element, "height").text = "480"
    ET.SubElement(size_element, "depth").text = "3"

    object_element = ET.SubElement(annotation, "object")
    ET.SubElement(object_element, "name").text = "CargoDoor"

    bndbox_element = ET.SubElement(object_element, "bndbox")
    ET.SubElement(bndbox_element, "xmin").text = str(xmin_from_csv)
    ET.SubElement(bndbox_element, "ymin").text = str(ymin_from_csv)
    ET.SubElement(bndbox_element, "xmax").text = str(xmax_from_csv)
    ET.SubElement(bndbox_element, "ymax").text = str(ymax_from_csv)

    ET.SubElement(object_element, "pose_x").text = str(pose_x_from_csv)
    ET.SubElement(object_element, "pose_y").text = str(pose_y_from_csv)
    ET.SubElement(object_element, "pose_z").text = str(pose_z_from_csv)

    tree = ET.ElementTree(annotation)
    xml_name = xml_write_path + str(int(filename[:-5]) + OFFSET) + ".xml"
    tree.write(xml_name)

'''###################################    USE WHAT YOU CAN    #######################################################'''
#     tree = ET.parse(path + filename)
#     root = tree.getroot()
#
#     for obj in root.iter('object'):  # loop through all the detected objects
#         if obj[0].text == 'CargoDoor':  # check if the detected object is a 'CargoDoor'
#             x = ET.SubElement(obj, 'pose_x')  # Add pose sub-elements
#             y = ET.SubElement(obj, 'pose_y')
#             z = ET.SubElement(obj, 'pose_z')
#
#     # Grab the row corresponding to the filename. No Donald.
#     try:
#         current_row = csv_data.iloc[int(filename[:-4])]
#     except ValueError:
#         print("\nExpected xml filename mismatch. Expecting a '<number>.xml', however got " + filename)
#
#     pose_x_from_csv = current_row.pose_x  # Assign pose data from the csv file
#     pose_y_from_csv = current_row.pose_y * 0
#     pose_z_from_csv = current_row.pose_z * 0
#
#     # pose_x_from_csv = np.random.uniform(0, 10)    # Assign pose data from the csv file
#     # pose_y_from_csv = np.random.uniform(2, 15)
#     # pose_z_from_csv = np.random.uniform(0, 0.5)
#
#     for obj in root.iter('object'):
#         if obj[0].text == 'CargoDoor':
#             obj[5].text = str(pose_x_from_csv)  # pose_x
#             obj[6].text = str(pose_y_from_csv)  # pose_y
#             obj[7].text = str(pose_z_from_csv)  # pose_z
#
#     tree.write(path + filename)
#
# print("Preprocessing done.")