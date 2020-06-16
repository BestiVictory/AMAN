import cv2
import numpy as np
import os
# from mobilenet import MobileNet
import sys
import time
from keras.models import load_model
from custom_layers.scale_layer import Scale


val_path = sys.argv[1]

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def writelistfile(list, filename):  
	with open(filename, 'a', encoding='utf8') as f:  
		for line in list:  
			f.write(line + '\n')

img_rows = 224
img_cols = 224
color_type = 3
weights_path = None
model = load_model('/home/graydove/Graydove/HMDNet/save_model/our_model_mobilenet.h5',custom_objects={'Scale': Scale})
# model = MobileNet()
result = []
result_str = ''


im = cv2.resize(cv2.imread(val_path), (224, 224)).astype(np.float32)
im = np.expand_dims(im, axis=0)
# model.compile()
out = model.predict(im)
for item in out:
	result_str += ' ' + '%.2f'%item[0][0]
result_str = result_str.strip(' ').split(' ')
result_str_10 = [10*float(i) for i in result_str]
#print result_str

print '[result]:%.1f,%.1f,%.1f,%.1f,%.1f,%.1f' % (float(result_str_10[1]), float(result_str_10[2]), float(result_str_10[3]), float(result_str_10[4]), float(result_str_10[5]), float(result_str_10[6]))
