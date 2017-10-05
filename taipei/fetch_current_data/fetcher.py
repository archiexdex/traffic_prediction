# python2
# environment: synology
import urllib
import gzip
import json
import os
import time
import datetime
import xml.etree.ElementTree as ET

url = "https://tcgbusfs.blob.core.windows.net/blobtisv/GetVDDATA.gz"
now = datetime.datetime.now()

ROOT_PATH = "/volume1/ShareData/DiskStation_001132127EB4/ShareData/Research related/DataBase/Taipei_current_data/"
SAVE_PATH = ROOT_PATH + "xml/"

new_minute = now.minute
old_minute = now.minute
current_data_time = ""
old_data_time = ""
while True:
    
    now = datetime.datetime.now()
    new_minute = now.minute

    if old_minute == new_minute:
        continue
   
    file_name = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
    try:
        url_opener = urllib.URLopener()
        url_opener.retrieve(url, ROOT_PATH + "tmp.gz")
    except:
        print("download data error!!")
        with open(ROOT_PATH + "log", "a") as fp:
            fp.write("[" + str(new_minute) + "] download data error!!\n")
        continue
    
    try:
        xml_file = gzip.open(ROOT_PATH + "tmp.gz", "rb")
        xml_data = xml_file.read()

        xml = gzip.open(ROOT_PATH + "tmp.gz", "rb")
        tree = ET.ElementTree(file=xml)
        for elem in tree.iter() :
            
            try:
                if elem.tag == "ExchangeTime":
                    current_data_time = datetime.datetime.strptime(elem.text, "%Y/%m/%dT%H:%M:%S").strftime("%Y%m%d_%H%M")
                    if current_data_time != old_data_time:
                        old_data_time = current_data_time

                        with open( ROOT_PATH + "xml/" + current_data_time + ".xml", "wb") as fp:
                            fp.write(xml_data)
                    break
            except:
                print("parse data error!!")
                with open( ROOT_PATH + "log", "a") as fp:
                    fp.write("[" + str(new_minute) + "] parse data error!!\n")
                break
    except:
        print("open tmp.gz error!!")
        with open(ROOT_PATH + "log", "a") as fp:
            fp.write("[" + str(new_minute) + "] open tmp.gz error!!\n")
    old_minute = new_minute

