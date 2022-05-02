# import the os module
import os

# detect the current working directory and print it
path = os.getcwd()
print("The current working directory is %s" % path)


path = "test"

try:
    os.mkdir(path)
except OSError:
    print ("Creation of the directory %s failed" % path)
else:
    print ("Successfully created the directory %s " % path)

path = "test2"

try:
    os.makedirs(path)
    print("Directory ", path, " Created ")
except FileExistsError:
    print("Directory ", path, " already exists")