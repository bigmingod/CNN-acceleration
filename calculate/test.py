#!/usr/bin/env python  
      
#_*_ coding:utf8 _*_

import caffe

import numpy as np

import sys

import struct

a = sys.argv[1]
#deploy file

b = sys.argv[2]
#caffemodel file

net = caffe.Net(str(a),str(b),caffe.TEST)

#object = open('para.bin','wb')

count_conv1 = 0
num_conv1 = 0
count_conv2 = 0
num_conv2 = 0

for param_name in net.params.keys():
	if param_name[0:5] != 'conv1':
		continue
	print param_name
	weight = net.params[param_name][0].data

	bias = net.params[param_name][1].data

	for w in weight:
		for i in w:	
			for j in i:
				#print j
				for k in j:
					num_conv1 = num_conv1 + 1
					if k == 0:
						count_conv1 = count_conv1 + 1

for param_name in net.params.keys():
	if param_name[0:5] != 'conv2':
		continue
	
	weight = net.params[param_name][0].data

	bias = net.params[param_name][1].data

	for w in weight:
		for i in w:	
			for j in i:
				for k in j:
					num_conv2 = num_conv2 + 1
					if k == 0:
						count_conv2 = count_conv2 + 1
count_ip1 = 0
num_ip1 = 0
count_ip2 = 0
num_ip2 = 0
	
for param_name in net.params.keys():
	if param_name[0:3] != 'ip1':
		continue
	
	weight = net.params[param_name][0].data
	for w in weight:
		for i in w:
			num_ip1 = num_ip1 + 1
			if i == 0:
				count_ip1 = count_ip1 +1

for param_name in net.params.keys():
	if param_name[0:3] != 'ip2':
		continue
	
	weight = net.params[param_name][0].data
	for w in weight:
		for i in w:
			num_ip2 = num_ip2 + 1
			if i == 0:
				count_ip2 = count_ip2 +1


conv1_w = net.params['conv1'][0].data

print conv1_w[0]
print conv1_w
print "conv1  num1"
print count_conv1,num_conv1
print "conv2 num2"
print count_conv2,num_conv2
print "ip1 num_ip1"
print count_ip1,num_ip1
print "ip2 num_ip2"
print count_ip2,num_ip2
#object.close()
