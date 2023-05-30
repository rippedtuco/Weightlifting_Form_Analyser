import cv2
import mediapipe as mp
import time
import numpy as np
import pandas as pd

class poseDetector():

    def __init__(self,
               static_image_mode=False,
               model_complexity=1,
               smooth_landmarks=True,
               enable_segmentation=False,
               smooth_segmentation=True,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5):
 
        self.static_image_mode= static_image_mode
        self.model_complexity=model_complexity
        self.smooth_landmarks=smooth_landmarks
        self.enable_segmentation=enable_segmentation
        self.smooth_segmentation=smooth_segmentation
        self.min_detection_confidence=min_detection_confidence
        self.min_tracking_confidence=min_tracking_confidence

        self.mpDraw=mp.solutions.drawing_utils
        self.mpPose=mp.solutions.pose
        self.pose= self.mpPose.Pose(self.static_image_mode,self.model_complexity,self.smooth_landmarks,self.enable_segmentation,
                                     self.smooth_segmentation, self.min_detection_confidence,self.min_tracking_confidence)
 
    
    def findPose(self, frames, draw=True):
        vidRGB=cv2.cvtColor(frames, cv2.COLOR_BGR2RGB)
        vidRGB.flags.writeable=False

        self.results=self.pose.process(vidRGB)

        vidRGB.flags.writeable=True
        vidRGB=cv2.cvtColor(frames, cv2.COLOR_RGB2BGR)
        

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(vidRGB,self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)

        return vidRGB

    def getPosition(self, frames, draw=True):
        lmList=[]
        for id,lm in enumerate(self.results.pose_landmarks.landmark):
            h,w,c=frames.shape
            cx,cy=int(lm.x * w),int(lm.y * h)
            lmList.append([id,cx,cy])
            if len(lmList)!=0:
                if draw:
                    cv2.circle(frames, (cx,cy), 5, (255,0,0), cv2.FILLED)
        return lmList

    def findAngleSP(self,frames):
        h,w,c=frames.shape
        self.results=self.pose.process(frames)
        lm= self.results.pose_landmarks.landmark

        r_shldr_x = int(lm[12].x * w)
        r_shldr_y = int(lm[12].y * h)

        r_elbow_x = int(lm[14].x * w)
        r_elbow_y = int(lm[14].y * h)

        r_wrist_x = int(lm[16].x * w)
        r_wrist_y = int(lm[16].y * h)

        r_angle= np.arctan2(r_elbow_y-r_shldr_y, r_elbow_x-r_shldr_x)-np.arctan2(r_elbow_y-r_wrist_y, r_elbow_x-r_wrist_x)
        r_angle = np.abs(r_angle*180.0/np.pi)

        if r_angle>180.0:
            r_angle=360-r_angle

        return r_angle

def main():

    vid=cv2.VideoCapture(r"videos\bi_right2.MOV")
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cv2.namedWindow("Frames", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Frames", 420, 420)
    pTime=0

    detectorx=poseDetector()
    r_angles_list = []

    while True:

        if(vid.isOpened()==False):
            print("Provide a Correct Video.")

        while (vid.isOpened()):
            success,frames=vid.read()

            frames=detectorx.findPose(frames)
            posObj=detectorx.getPosition(frames, draw=False)
            r_angle=detectorx.findAngleSP(frames)
            r_angles_list.append(r_angle)
            cv2.circle(frames, (posObj[25][1],posObj[25][2]), 5, (0,0,255), cv2.FILLED)

            cTime=time.time()
            fps=1/(cTime-pTime)
            pTime=cTime

            cv2.putText(frames, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
            

            if success==True:
               cv2.imshow("Frames",frames)
               cv2.waitKey(1)
            else:
                break
            #print(r_angles_list)
            df = pd.DataFrame({'bi_right2' :  r_angles_list})
            df.to_csv('RAL.csv',index=False)

        
if __name__ == "__main__":
    main()