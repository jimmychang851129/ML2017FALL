import sys
from PIL import Image
im = Image.open(sys.argv[1])
img_size = im.size
rgb_im = im.convert('RGB')

pixel = im.load()
for i in range(img_size[0]):
		for j in range(img_size[1]):
			r, g, b = rgb_im.getpixel((i, j))
			pixel[i,j]=(r//2,g//2,b//2)
im.save('Q2.png')
