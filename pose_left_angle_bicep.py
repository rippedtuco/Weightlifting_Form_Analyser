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

        l_shldr_x = int(lm[11].x * w)
        l_shldr_y = int(lm[11].y * h)

        l_elbow_x = int(lm[13].x * w)
        l_elbow_y = int(lm[13].y * h)

        l_wrist_x = int(lm[15].x * w)
        l_wrist_y = int(lm[15].y * h)

        r_mouth_r_x = int(lm[9].x * w)
        r_mouth_r_y = int(lm[9].y * w)

        r_hip_r_x = int(lm[23].x * w)
        r_hip_r_y = int(lm[23].y * w)

        b_angle=np.arctan2(r_mouth_r_y-r_hip_r_y, r_mouth_r_x-r_hip_r_x)
        b_angle = np.abs(b_angle*180.0/np.pi)

        l_angle= np.arctan2(l_elbow_y-l_shldr_y, l_elbow_x-l_shldr_x)-np.arctan2(l_elbow_y-l_wrist_y, l_elbow_x-l_wrist_x)
        l_angle = np.abs(l_angle*180.0/np.pi)

        if l_angle>180.0:
            l_angle=360-l_angle

        return l_angle, b_angle

def main():

    vid=cv2.VideoCapture(r"videos\bi_right2.MOV")
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cv2.namedWindow("Frames", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Frames", 420, 420)
    pTime=0

    detectorx=poseDetector()
    r_angles_list = []
    b_angles_list = []

    while True:

        if(vid.isOpened()==False):
            print("Provide a Correct Video.")

        while (vid.isOpened()):
            success,frames=vid.read()

            frames=detectorx.findPose(frames)
            posObj=detectorx.getPosition(frames, draw=False)
            angObj=detectorx.findAngleSP(frames)
            r_angles_list.append(angObj[0])
            b_angles_list.append(angObj[1])
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
            # headers = ["elbow_angle","back_angle"]
            # df.columns=headers
            df = pd.DataFrame({'elbow_angle' :  r_angles_list,'back_angle' : b_angles_list})
            df.to_csv('RAL.csv',index=False, header=True)

        
if __name__ == "__main__":
    main()
