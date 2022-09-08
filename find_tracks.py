import numpy as np
import matplotlib.pyplot as plt
import cv2
from feature import *


def find_windows(p):
    f1,f2, ret = features(p)
    return ret, f1
    

def find_tracks(f1, f2):
    windows = {}
    p = '../cv_timepass/frames/frame'
    windowed_images = []
    allWin = []
    for i in range(f1,f2+1):
        win, im = find_windows(p+str(i)+'.jpg')
        allWin.append(win)
        windowed_images.append(im)
        windows[i] = win
   #     print(win)
        #plt.imshow(im, cmap='gray')
        #plt.show()

    print('Found windows in all frames !')
    
    lk_params = dict( winSize = (15, 15),maxLevel = 6,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10, 0.03))
    lk_pos =[] 
    image_dim = (1280,720)
    po = [[i,j] for i in range(image_dim[0]) for j in range(image_dim[1])]
    po = np.array(po).astype(np.float32)
    for frame in range(f1,f2):
        im1 = cv2.imread(p+str(frame)+'.jpg', cv2.IMREAD_GRAYSCALE)
        im2 = cv2.imread(p+str(frame+1)+'.jpg', cv2.IMREAD_GRAYSCALE)
        nex, r1, r2 = cv2.calcOpticalFlowPyrLK(im1, im2,po, None, **lk_params)
        lk_pos.append(nex)
    print('Done till here .')
    graph = []
    for i in range(f1, f2):
        im1 = cv2.imread(p+str(frame)+'.jpg', cv2.IMREAD_GRAYSCALE)
        #plt.imshow(im1, cmap='gray')
        #plt.show()
        im2 = cv2.imread(p+str(frame)+'.jpg', cv2.IMREAD_GRAYSCALE)
        for win in windows[i]:
            cur = np.array([[i,j] for i in range(win[0],win[2]+1) for j in range(win[1],win[3]+1)])
            for j in range(i+1,f2+1):
                done = 0
                co = np.zeros(len(windows[j]))
                for l1 in range(cur.shape[0]):
                    x,y = int(cur[l1][0]), int(cur[l1][1])
                    if x>=0 and x<image_dim[0] and y>=0 and y<=image_dim[1]:
                        cur[l1][0] = lk_pos[j-f1-1][x*image_dim[1]+y][0]
                        cur[l1][1] = lk_pos[j-f1-1][x*image_dim[1]+y][1]
                        for wii in range(len(windows[j])):
                            wi = windows[j][wii]
                            if cur[l1][0]>=wi[0] and cur[l1][0]<=wi[2] and cur[l1][1]>=wi[1] and cur[l1][1]<=wi[3]:
                                co[wii]+=1
                if co.shape[0]>0:
                    print(co.max())
                for wii in range(len(windows[j])):
                    if co[wii]>=0.5*cur.shape[0]:
                        graph.append((win,windows[j][wii],i,j))
                        done = 1
                        break
                if done == 1:
                    break
        print('Done for : ', i)
    return graph, lk_pos, windowed_images, allWin    

#graph, lk_pos = find_tracks(697, 718)
