import numpy as np
import cv2
import os

def read_frames(path):
    cap = cv2.VideoCapture(path)
#    print('yo')
    frames = []
    count = 0
    while cap.isOpened() and count<600:
        cap.set(cv2.CAP_PROP_POS_MSEC,count*100)
        ret, frame = cap.read()
        if ret == True:
            count+=1
            cv2.imwrite('./frames/frame'+str(count)+'.jpg', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if count%100 == 0:
            print('Done till -- ', count)

#read_frames('ep1.mkv')

def check_hist(p1, p2, threshold_factor = 2e5):
    im1 = cv2.imread(p1, cv2.IMREAD_UNCHANGED)
    im2 = cv2.imread(p2, cv2.IMREAD_UNCHANGED)
    bins = 256
    ran = (0,256)
    h1 = []
    h2 = []
    h1.append(np.histogram(im1[:,:,0], bins, range=ran)[0])
    h1.append(np.histogram(im1[:,:,1], bins, range=ran)[0])
    h1.append(np.histogram(im1[:,:,2], bins, range=ran)[0])
    h2.append(np.histogram(im2[:,:,0], bins, range=ran)[0])
    h2.append(np.histogram(im2[:,:,1], bins, range=ran)[0])
    h2.append(np.histogram(im2[:,:,2], bins, range=ran)[0])
    h1 = np.array(h1)
    h2 = np.array(h2)
    dif = (np.abs(h1-h2)).sum()
    # print('Between p1 and p2  : ', dif)
    if dif<threshold_factor:
        return 1
    return 0


def find_shots(path):
    files = [f for f in os.listdir(path)]
    frame_count = len(files)
    edge_flag  = []
    for i in range(1, frame_count):
        fl=check_hist(path+'/frame'+str(i)+'.jpg',path+'/frame'+str(i+1)+'.jpg',threshold_factor=1e5)
        if i%100 == 0:
            print('Done till -- ', i)
        edge_flag.append(fl)

    i = 0
    shots = []
    while i<frame_count:
        cur = [i+1]
        j = i
        while j<frame_count-1 and edge_flag[j] == 1:
            cur.append(j+2)
            j+=1
        i = j
        i+=1
        shots.append(cur)
    
    return shots

#print(find_shots('./frames'))

