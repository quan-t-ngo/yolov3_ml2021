import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
from PIL import Image
from os import path, makedirs
import re
import sys
import argparse

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

def convert_csv_to_yolo(multi_df, labeldict, path, target_name):

    # Encode labels according to labeldict if code's don't exist
    if not "code" in multi_df.columns:
        multi_df["code"] = multi_df["class"].apply(lambda x: labeldict[x])
    # Round float to ints
    for col in multi_df[["xmin", "ymin", "xmax", "ymax"]]:
        multi_df[col] = (multi_df[col]).apply(lambda x: round(x))

    # Create Yolo Text file
    last_image =""
    txt_file = ""

    for index, row in multi_df.iterrows():
        if not last_image == row["filename"]:
            txt_file += "\n" + path + row["filename"] + " "
            txt_file += ",".join(
                [
                    str(x)
                    for x in (row[["xmin", "ymin", "xmax", "ymax", "code"]].tolist())
                ]
            )
        else:
            txt_file += " "
            txt_file += ",".join(
                [
                    str(x)
                    for x in (row[["xmin", "ymin", "xmax", "ymax", "code"]].tolist())
                ]
            )
        last_image = row["filename"]
    file = open(target_name, "w")
    file.write(txt_file[1:])
    file.close()
    return True


if __name__ == "__main__":
    # surpress any inhereted default values
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    """
    Command line options
    """
    
    parser.add_argument(
        "VOC_xml_folder_path",
        type=str,
		default="./annotations/",
        help="Absolute path to the VOC format annotation folder. Default is: ./annotations/",
    )
    
    parser.add_argument(
        "Image_folder_path",
        type=str,
		default="./images/",
        help="Absolute path to the image folder. Default is: ./images/",
    )

    parser.add_argument(
        "YOLO_txt_filename",
        type=str,
		default='yolo_label.txt',
        help="Absolute path to the file where the annotations in YOLO format should be saved. Default is: yolo_label.txt",
    )

    FLAGS = parser.parse_args()

    # Prepare the dataset for YOLO
#     multi_df = pd.read_csv(FLAGS.VoTT_csv)
    
    voc_xml_path = FLAGS.VOC_xml_folder_path
    multi_df = xml_to_csv(voc_xml_path)
    
    labels = multi_df["class"].unique()
    labeldict = dict(zip(labels, range(len(labels))))
    multi_df.drop_duplicates(subset=None, keep="first", inplace=True)
    image_path = FLAGS.Image_folder_path
    convert_csv_to_yolo(
        multi_df, labeldict, path=image_path, target_name=FLAGS.YOLO_txt_filename
    )

    # Make classes file
    classes_filename = os.path.join(voc_xml_path, "data_classes.txt")

    file = open(classes_filename, "w")

    # Sort Dict by Values
    SortedLabelDict = sorted(labeldict.items(), key=lambda x: x[1])
    for elem in SortedLabelDict:
        file.write(elem[0] + "\n")
    file.close()

