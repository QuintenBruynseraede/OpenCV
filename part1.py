
import cv2
import numpy as np
import time    #time.sleep()

cap = cv2.VideoCapture('Video/petanque3_goodlighting.AVI')  
cap2 = cv2.VideoCapture('Video/multipleballs_goodlighting.AVI')
active_video = cap
if (cap.isOpened()== False):
    print("Error opening video stream or file")

fps = cap.get(cv2.CAP_PROP_FPS)
start_sec = 0
current_frame = fps * start_sec
cap.set(1,current_frame)
font = cv2.FONT_HERSHEY_SIMPLEX
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
out = cv2.VideoWriter('part1.avi',fourcc, 20, (round(frame_width),round(frame_height)))

def printSubtitle(frame,text):
    textSize = cv2.getTextSize(text,font,0.6,2)[0]
    textX = (frame.shape[1] - textSize[0]) // 2
    textY = (frame.shape[0] + textSize[1]) // 2 + 200
    cv2.putText(frame,text,(textX,textY),font,0.6, (255,255,255), 1)
    
def isIn(sec,lower,upper):
    return lower <= sec < upper

def printSubject(frame,text):
    frame = cv2.rectangle(frame,(frame.shape[0]-35,23),(frame.shape[0]+200,65),(0,0,0),-1)
    cv2.putText(frame,text,(frame.shape[0]-30,50),font,0.6, (255,255,255), 1)

while(cap.isOpened()):
    ret, frame = active_video.read()
    current_frame += 1
    sec = current_frame/fps
    
    if sec > 20:
        break

    if isIn(sec, 0,4):
        '''Switch the movie between color and grayscale a few times'''
        if isIn(sec,1,2):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif isIn(sec,3,4):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        printSubject(frame,"Black and white")
            
    elif isIn(sec,4,12):
        
        '''Experiment with Gaussian and bi-lateral filters and increase the 
        effect by widening your filter kernel. Clearly explain the difference 
        between the two in the subtitles'''
        if isIn(sec,4,8):
            '''Gaussian filter 1,3,5,'''
            kernel_size = (1,1)
            if isIn(sec,5,8):
                kernel_size = (round((1 + ((sec-5)/3)*100)/2)*2 + 1,round((1 + ((sec-5)/3)*100)/2)*2 + 1)
            frame = cv2.GaussianBlur(frame,kernel_size,cv2.BORDER_DEFAULT)
            printSubtitle(frame,"Kernel size "+str(kernel_size[0])+","+str(kernel_size[1]))
            printSubject(frame,'Gaussian blur')
            
        if isIn(sec,8,12):
            '''Bilateral filter'''
            if isIn(sec,8,9):
                printSubtitle(frame,"Bilateral filter OFF")
            if isIn(sec,9, 10):
                printSubtitle(frame,"Bilateral filter ON. Filter size = 10. Sigma = (25,25)")
                frame = cv2.bilateralFilter(frame,10,25,25)
            if isIn(sec,10,11):
                printSubtitle(frame,"Bilateral filter ON. Filter size = 20. Sigma = (25,25)")
                frame = cv2.bilateralFilter(frame,20,25,25)
            if isIn(sec,11,12):
                printSubtitle(frame,"Bilateral filter ON. Filter size = 40. Sigma = (25,25)")
                frame = cv2.bilateralFilter(frame,40,25,25)                
            printSubject(frame,'Bilateral filter')

        
    elif isIn(sec, 12,20):
        active_video = cap2
        if isIn(sec, 13,14.5):
            red_lower = np.array([0,0,85])
            red_upper = np.array([70,70,255])
            frame = cv2.inRange(frame,red_lower,red_upper)
            #frame = cv2.bitwise_and(frame,frame,mask=mask)
            printSubtitle(frame,"Grabbing red ball in BGR space.")
        if isIn(sec,14.5,16.5):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            red_lower = np.array([172,180,0])
            red_upper = np.array([179,255,255])
            mask = cv2.inRange(frame,red_lower,red_upper)
            frame = mask
            printSubtitle(frame, "Grabbing red ball in HSV space.")

        if isIn(sec, 16.5,20):
            '''Improved grabbing: binary morphological operations'''
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            red_lower = np.array([168,130,50])
            red_upper = np.array([179,255,255])
            mask_basic = cv2.inRange(frame,red_lower,red_upper)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(25,25))
            mask_improved = cv2.morphologyEx(mask_basic,cv2.MORPH_CLOSE, kernel)
            kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))
            mask_improved = cv2.erode(mask_improved, kernel2, iterations = 1)
            
            mask_basic_n = cv2.bitwise_not(mask_basic)
            mask_diff = cv2.bitwise_and(mask_improved,mask_improved,mask=mask_basic_n)
            green = np.zeros((mask.shape[0], mask.shape[1], 3), np.uint8)
            green[:] = (0, 255, 0)
            mask_diff_green = cv2.bitwise_and(green,green,mask=mask_diff)
            
            mask_improved_n = cv2.bitwise_not(mask_improved)
            mask_diff2 = cv2.bitwise_and(mask_basic,mask_basic,mask=mask_improved_n)
            red = np.zeros((mask.shape[0], mask.shape[1], 3), np.uint8)
            red[:] = (0, 0, 255)
            mask_diff_red = cv2.bitwise_and(red,red,mask=mask_diff2)
            
            mask_total = cv2.bitwise_or(cv2.cvtColor(mask_basic,cv2.COLOR_GRAY2RGB),mask_diff_green)
            mask_total2 = cv2.add(mask_total,mask_diff_red)
            frame = mask_total

            printSubtitle(frame, "Improved grabbing: closing + erosion.")
        printSubject(frame,"Grabbing")

                    
    cv2.putText(frame,"Frame "+str(current_frame), (10,30),font,0.5,(255,255,255),1)
    cv2.putText(frame,"Sec "+str(round(sec,2)), (10,50), font, 0.5, (255,255,255), 1)
    out.write(frame)
    
    if ret == True:
        cv2.imshow('Frame',frame)
    
    key = cv2.waitKey(int(1000/fps)) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        cv2.waitKey(-1)

cap.release()
out.release()
cv2.destroyAllWindows()
