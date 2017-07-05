
import os
import csv
import sys

root_path = "/home/xdex/Documents/VD/201501-201703/"

file = open(root_path + "20151212000000_20160112000000.csv")
csv_cursor = csv.reader(file)

for i in csv_cursor:
    print (i)
    input("!!")


    