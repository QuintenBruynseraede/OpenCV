#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np

cap = cv2.VideoCapture('Video/petanque2_goodlighting.AVI')  
cap2 = cv2.VideoCapture('Video/petanque2_goodlighting_cut.AVI')

active_video = cap
if (cap.isOpened()== False):
    print("Error opening video stream or file")

fps = cap.get(cv2.CAP_PROP_FPS)
start_sec = 0
current_frame = fps * start_sec
cap.set(1,current_frame)
font = cv2.FONT_HERSHEY_SIMPLEX
frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
out = cv2.VideoWriter('part2.avi',fourcc, 20, (round(frame_width),round(frame_height)))


def printSubtitle(frame,text):
    textSize = cv2.getTextSize(text,font,0.6,2)[0]
    textX = (frame.shape[1] - textSize[0]) // 2
    textY = (frame.shape[0] + textSize[1]) // 2 + 200
    cv2.putText(frame,text,(textX,textY),font,0.6, (255,255,255), 1)
    
def printSubject(frame,text):
    frame = cv2.rectangle(frame,(frame.shape[0]-35,23),(frame.shape[0]+200,65),(0,0,0),-1)
    cv2.putText(frame,text,(frame.shape[0]-30,50),font,0.6, (255,255,255), 1)
    
def isIn(sec,lower,upper):
    return lower <= sec < upper


while(cap.isOpened()):
    ret, frame = active_video.read()
    current_frame += 1
    sec = current_frame/fps
    k_size = 1
    
    '''Sobel Edge detection'''
    if isIn(sec,0,5):
        if isIn(sec,0.8,1.5):
            k_size = 1
        elif isIn(sec,1.5,2.2):
            k_size = 11
        elif isIn(sec,2.2,3.6):
            k_size = 15
        elif isIn(sec,3.6,4.2):
            k_size = 19
        elif isIn(sec,4.2,5): 
             k_size = 21

        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        sobely = cv2.Sobel(frame,cv2.CV_8U,0,1,ksize=k_size)
        sobelx = cv2.Sobel(frame,cv2.CV_8U,1,0,ksize=k_size)
        edges = cv2.add(sobelx,sobely)
        
        green = np.zeros((edges.shape[0], edges.shape[1], 3), np.uint8)
        green[:] = (0, 255, 0)
        red = np.zeros((edges.shape[0], edges.shape[1], 3), np.uint8)
        red[:] = (0, 0, 255)

        #Vertical edges in green
        edge_mask_green = cv2.bitwise_and(green,green,mask=sobelx)
        #Horizontal edges in red
        edge_mask_red = cv2.bitwise_and(red,red,mask=sobely)
        frame = cv2.add(edge_mask_green,edge_mask_red)
        #frame = edge_mask_green
        #frame = edge_mask_red

        frame = cv2.rectangle(frame,(0,425),(640,453),(0,0,0),-1)
        printSubtitle(frame,"Filter size={}. Green = vertical, red = horizontal".format(k_size))
        printSubject(frame,"Edge detection")
        
    if isIn(sec,5,15):
        min_radius = 0
        max_radius = 100
        d_t = 5.95
        '''Circle detection '''
        if isIn(sec,6,7):
            min_radius = 10
        elif isIn(sec,7,8):
            min_radius = 10
            max_radius = round(100-(sec-7)*60)
        elif isIn(sec,8,10):
            min_radius = 10
            max_radius = 40
            d_t = round(5.95-(sec-8)/2*4,2)
        elif isIn(sec,10,15):
            min_radius = 10
            max_radius = 40
            d_t = 1.95
        
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, d_t, 5,minRadius=min_radius,maxRadius=max_radius,param1=100,param2=99)
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            print(circles)
            for (x, y, r) in circles:      
                cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
            
        printSubject(frame,"Circle detection")
        printSubtitle(frame,'min/max radius: {},{}, dt: {} using GRADIENT)'.format(min_radius,max_radius,d_t))
        #frame = gray
    
    if isIn(sec,15,20):
        active_video = cap2
        if isIn(sec,15,15.1):
            #Create a template for the small pink ball
            template_x = 313
            template_y = 228
            template_w = 45
            template_h = 42
            template = frame[template_y:template_y+template_h,template_x:template_x+template_w]
        if isIn(sec,15,17):
            #Draw static rectangle around template
            if round(sec*6) % 2 == 0:
                cv2.rectangle(frame,(template_x,template_y),(template_x+template_w,template_y+template_h),(0,255,0),thickness=2)
        elif isIn(sec,17,20):
            found = cv2.matchTemplate(frame,template,cv2.TM_SQDIFF_NORMED)
            _, _, min_loc, max_loc = cv2.minMaxLoc(found)
            one = np.ones((439,596)).astype(np.float32)
            frame = cv2.cvtColor(one-found,cv2.COLOR_GRAY2BGR)
            
            cv2.rectangle(frame,min_loc,(min_loc[0]+template_w,min_loc[1]+template_h),(0,255,0),thickness=2)
            
        printSubject(frame,"Template matching")

           
    cv2.putText(frame,"Frame "+str(current_frame), (10,30),font,0.5,(255,255,255),1)
    cv2.putText(frame,"Sec "+str(round(sec,2)), (10,50), font, 0.5, (255,255,255), 1)
    
    if ret == True:
        cv2.imshow('Frame',frame)
        out.write(frame.astype(np.uint8))
    
    key = cv2.waitKey(int(1000/fps)) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        cv2.waitKey(-1)

cap.release()
out.release()
cv2.destroyAllWindows()
