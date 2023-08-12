import os 
import cv2
import glob 

path="./trial_id1/pretrained.pth.tar"
size=(512,512)
out = cv2.VideoWriter('video_id1.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 30,size)
img_array=[]
for i in range(500):
    img=cv2.imread(os.path.join(path,f"%04d"%(i)+".png"))
    print(i)
    img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    height, width, _ = img.shape
    out.write(img)
out.release()
