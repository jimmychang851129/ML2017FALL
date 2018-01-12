import numpy as np
import skimage.io
from skimage import transform
from skimage import io
import sys

a = []
cnt = 415
scale = 
reconstruct = [50,100,150,200]
tmp_construct = 50
allmean = 0
eigenface_cnt = 4
file_prefix = sys.argv[1]
for i in range(cnt):
    file = file_prefix+str(i)+".jpg"
    img = skimage.io.imread(file)
    img = transform.resize(img,(scale,scale,3))
    a.append(img.flatten())

a = np.array(a)
mean = np.mean(a,axis=0)
a -= mean

U, s, V = np.linalg.svd(a, full_matrices=False)

##############
# recontruct #
##############
eigenface = V[:eigenface_cnt,:]

file = sys.argv[2]

img = skimage.io.imread(file)
img = transform.resize(img,(scale,scale,3))
img = img.flatten()
img -= mean[tmp_construct]
img = img.reshape(scale*scale*3,1)
w = np.dot(eigenface,img)

output = np.dot(w.T,eigenface)
output = output.reshape(270000,1)


output += mean[tmp_construct]
output -= np.min(output)
output /= np.max(output)
output=(output*255).astype(np.uint8)

output = output.reshape(scale,scale,3)

io.imsave("./reconstruction.jpg",output)