import cv2
import PIL.Image, PIL.ImageTk
from PIL import Image

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1200)
    i = 317
    while True:
        success, img = cap.read()
        img1 = cv2.flip(img,1)
        imgRGB = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
        
        cv2.rectangle(img,(0,0),(60,1200),(255,255,255),-1)
        cv2.rectangle(img,(1200,0),(1600,1200),(255,255,255),-1)
        cv2.rectangle(img,(0,0),(1600,40),(255,255,255),-1)
        cv2.rectangle(img,(0,650),(1600,1200),(255,255,255),-1)
        cv2.imshow("video", img)
        key = cv2.waitKey(33)
        if key == 27:
            break
        elif key == 26:  # ctrl + z
            print("Capture Success!!!")
            cv2.IMREAD_UNCHANGED
            cv2.imwrite("./data/train/non/"+str(i)+ ".png", img)
            i +=1
    cap.release()
    cv2.destroyAllWindows()