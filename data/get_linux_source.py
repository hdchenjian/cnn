#!/usr/bin/python3
# encoding: utf-8

import commands

dir_path = 'linux/'

status, output = commands.getstatusoutput('find ' + dir_path + ' -name *.c')
print(status)#
#print(output.split('\n'))
all_code = output.split('\n')

d = open('linux.txt', 'w')
for _code in all_code:
    f = open(_code, 'rU')
    c = f.read()
    d.write(c + '\n\n\n\n\n')
    f.close()
d.close()
