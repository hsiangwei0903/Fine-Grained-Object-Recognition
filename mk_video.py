import glob
import os

imgs_path = glob.glob('/home/wei/Desktop/wyze/*')
out_path = '/home/wei/Desktop/output'
for img in imgs_path:
    name = img.replace('/home/wei/Desktop/wyze/','')
    cmd_str = 'ffmpeg -f image2 -i {} -c:v copy {}'.format(img+'/'+'%04d'+'.jpg',out_path+'/'+name+'.mp4')
    os.system(cmd_str)