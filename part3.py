#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import math

cap = cv2.VideoCapture('Video/petanque2_goodlighting.AVI')  

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
out = cv2.VideoWriter('part3.avi',fourcc, 20, (round(frame_width),round(frame_height)))

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

def dist(x,y):
    return math.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)


while(cap.isOpened()):
    ret, frame = active_video.read()
    if (frame is None):
        break
    current_frame += 1
    sec = current_frame/fps

    template_pink = cv2.imread("Part3_templates/pink.png")
    template_green = cv2.imread("Part3_templates/green.png")
    template_red = cv2.imread("Part3_templates/red.png")
    template_blue = cv2.imread("Part3_templates/blue.png")
                
    
    if isIn(sec,0,20):
        min_threshold = 0.15
        if isIn(sec,0,3):
            printSubtitle(frame,"Let's play some 2D petanque")
        elif isIn(sec,3,6):
            printSubtitle(frame,"Balls are detected using template matching.")
        elif isIn(sec,6,9):
            printSubtitle(frame,"The current winning ball is indicated.")
        elif isIn(sec,9,12):
            printSubtitle(frame,"The crown has had its background removed.")
                        
        found_pink = cv2.matchTemplate(frame,template_pink,cv2.TM_SQDIFF_NORMED)
        found_green = cv2.matchTemplate(frame,template_green,cv2.TM_SQDIFF_NORMED)
        found_red = cv2.matchTemplate(frame,template_red,cv2.TM_SQDIFF_NORMED)
        found_blue = cv2.matchTemplate(frame,template_blue,cv2.TM_SQDIFF_NORMED)
        
        min_pink, _, loc_pink, _ = cv2.minMaxLoc(found_pink)
        min_green, _, loc_green, _ = cv2.minMaxLoc(found_green)
        min_red, _, loc_red, _ = cv2.minMaxLoc(found_red)
        min_blue, _, loc_blue, _ = cv2.minMaxLoc(found_blue)
        
        center_pink = (round(loc_pink[0]+template_pink.shape[0]/2),round(loc_pink[1]+template_pink.shape[1]/2))
        center_green = (round(loc_green[0]+template_green.shape[0]/2),round(loc_green[1]+template_green.shape[1]/2))
        center_blue = (round(loc_blue[0]+template_blue.shape[0]/2),round(loc_blue[1]+template_blue.shape[1]/2))
        center_red = (round(loc_red[0]+template_red.shape[0]/2),round(loc_red[1]+template_red.shape[1]/2))
        
        dist_red = dist(center_pink,center_red)
        dist_green = dist(center_pink,center_green)
        dist_blue = dist(center_pink,center_blue)
        dist_min = min(dist_red,dist_green,dist_blue)
        
        crown = cv2.imread("crown.png")
        crown_mask = cv2.bitwise_not(cv2.inRange(crown,(0,0,250),(255,255,255)))
        zero = np.zeros((70,70,3),dtype=np.uint8)
        crown = cv2.bitwise_or(zero,crown,mask=crown_mask)

        
        if min_pink < min_threshold:
            cv2.rectangle(frame,loc_pink,(loc_pink[0]+template_pink.shape[0],loc_pink[1]+template_pink.shape[1]),(203,192,255))
        if min_green < min_threshold:
            cv2.line(frame,center_pink,center_green,(0,255,0))
            cv2.rectangle(frame,loc_green,(loc_green[0]+template_green.shape[0],loc_green[1]+template_green.shape[1]),(0,255,0))
        if min_blue < min_threshold:
            cv2.line(frame,center_pink,center_blue,(255,0,0))
            cv2.rectangle(frame,loc_blue,(loc_blue[0]+template_blue.shape[0],loc_blue[1]+template_blue.shape[1]),(255,0,0))
        if min_red < min_threshold:
            cv2.line(frame,center_pink,center_red,(0,0,255))
            cv2.rectangle(frame,loc_red,(loc_red[0]+template_red.shape[0],loc_red[1]+template_red.shape[1]),(0,0,255))

        if isIn(sec,6,20):
            if dist_min == dist_red:
                x1 = loc_red[0]-8
                y1 = loc_red[1]-57
            elif dist_min == dist_blue:
                x1 = loc_blue[0]-8
                y1 = loc_red[1]-57
            elif dist_min == dist_green:
                x1 = loc_green[0]-8
                y1 = loc_green[1]-57
                
            r = crown.shape[1]
            c = crown.shape[0]
                
            for ir in range(r):
                for ic in range(c):
                    if crown_mask[ir,ic] == 255:
                        frame[y1+ir,x1+ic] = crown[ir,ic]

            
    printSubject(frame,"Carte Blanche")
    cv2.putText(frame,"Frame "+str(current_frame), (10,30),font,0.5,(255,255,255),1)
    cv2.putText(frame,"Sec "+str(round(sec,2)), (10,50), font, 0.5, (255,255,255), 1)
    
    if ret == True:
        out.write(frame)
        cv2.imshow('Frame',frame)
    
    key = cv2.waitKey(int(1000/fps)) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        cv2.waitKey(-1)

cap.release()
out.release()
cv2.destroyAllWindows()
