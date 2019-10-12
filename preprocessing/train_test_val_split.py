import os
import re
import sys
import glob
import random

random.seed(3)

os.chdir(sys.argv[1])
dirs = [dir for dir in os.listdir()]
length = 0
os.mkdir("train")
os.mkdir("val")
os.mkdir("test")
to_skip = ["train", "test", "val"]
debug = False

for dir in dirs:
    if dir in to_skip:
        continue
    os.mkdir("train/"+dir)
    os.mkdir("test/"+dir)
    os.mkdir("val/"+dir)
    os.chdir(dir)
    files = os.listdir()
    random.shuffle(files) 
    files = files[0:80]

    length = len(files)

    train_l = int(length * 0.7)
    val_l = train_l + int(length * 0.1)

    print(dir, length, train_l, val_l)

    if debug:
        print()
        print("Folder name:", dir)
        print("Total length:", length)
        print("Train stopping point:", train_l)
        print("Validation stopping point:", int(length * 0.1), val_l)

    for ind, img in enumerate(files):
        if ind <= train_l:
            os.rename(img, "../train/"+dir+"/"+img)
        elif ind <= val_l:
            os.rename(img, "../val/"+dir+"/"+img)
        else:
            os.rename(img, "../test/"+dir+"/"+img)
    os.chdir("..")

os.chdir("..")
