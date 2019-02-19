# !/usr/bin/python
# coding:utf8

f = open('test')
lines = f.readlines()
for i, line in enumerate(lines):
    print(i, line.strip())
f.close()