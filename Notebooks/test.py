file = open("exp_data-2017-4-14-18-36.dict")
file.encoding
import ast
line = file.readline()
# python2 had separate types for int and long; python3 does not
line = line.replace("L", "")
line = line.replace("nan", "False")

print(type(line))

print("58687864160L" in line)
sepline = eval(line)
print(type(sepline))
print(sepline['viewPos_XYZ'])
