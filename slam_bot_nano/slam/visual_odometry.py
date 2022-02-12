import numpy as np
import cv2

STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2
kMinNumFeature = 1500

lk_params = dict(winSize  = (21, 21),
#maxLevel = 3,
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

def featureTracking(image_ref, image_cur, px_ref):
    if px_ref.size == 0:
        return px_ref, np.array([])

    kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, **lk_params)  #shape: [k,2] [k,1] [k,1]

    if st is None:
        return px_ref, np.array([])

    st = st.reshape(st.shape[0])
    kp1 = px_ref[st == 1]
    kp2 = kp2[st == 1]

    return kp1, kp2

def drawFeatureTracks(img, kps_ref, kps_cur, mask_match):
    draw_img = img
    num_outliers = 0

    for i,pts in enumerate(zip(kps_ref, kps_cur)):
        if mask_match[i]:
            p1, p2 = pts
            a,b = p1.astype(int).ravel()
            c,d = p2.astype(int).ravel()
            cv2.line(draw_img, (a,b),(c,d), (0,255,0), 1)
            cv2.circle(draw_img,(a,b),1, (0,0,255),-1)
        else:
            num_outliers+=1

    return draw_img

class VisualOdometry:
    def __init__(self, cam):
        self.frame_stage = 0
        self.cam = cam
        self.new_frame = None
        self.last_frame = None
        self.cur_R = None
        self.cur_t = None
        self.px_ref = None
        self.px_cur = None
        self.focal = cam.fx
        self.pp = (cam.cx, cam.cy)
        self.trueX, self.trueY, self.trueZ = 0, 0, 0
        self.detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
        self.matched_image = None

    def getAbsoluteScale(self, frame_id):  #specialized for KITTI odometry dataset
        return float(1)

    def processFirstFrame(self):
        self.px_ref = self.detector.detect(self.new_frame)
        self.px_ref = np.array([x.pt for x in self.px_ref], dtype=np.float32)
        self.frame_stage = STAGE_SECOND_FRAME

    def processSecondFrame(self):
        self.px_ref, self.px_cur = featureTracking(self.last_frame, self.new_frame, self.px_ref)
        E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, self.cur_R, self.cur_t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal, pp = self.pp)
        self.frame_stage = STAGE_DEFAULT_FRAME
        self.px_ref = self.px_cur

    def processFrame(self, frame_id):
        self.px_ref, self.px_cur = featureTracking(self.last_frame, self.new_frame, self.px_ref)

        E = None

        if self.px_ref.size != 0 and self.px_cur.size != 0:
            E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            if mask is not None:
                self.matched_image = drawFeatureTracks(self.colour_img, self.px_ref, self.px_cur, mask)
        else:
            self.matched_image = None

        if E is not None and E.shape[0] == 3 and E.shape[1] == 3:
            _, R, t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal, pp = self.pp)
            absolute_scale = self.getAbsoluteScale(frame_id)
            if(absolute_scale > 0.1):
                self.cur_t = self.cur_t + absolute_scale*self.cur_R.dot(t)
                self.cur_R = R.dot(self.cur_R)

        if(self.px_ref.shape[0] < kMinNumFeature):
            self.px_cur = self.detector.detect(self.new_frame)
            self.px_cur = np.array([x.pt for x in self.px_cur], dtype=np.float32)
        self.px_ref = self.px_cur

    def update(self, img, frame_id):
        self.colour_img = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        assert(img.ndim==2 and img.shape[0]==self.cam.height and img.shape[1]==self.cam.width), "Frame: provided image has not the same size as the camera model or image is not grayscale"
        self.new_frame = img
        if(self.frame_stage == STAGE_DEFAULT_FRAME):
            self.processFrame(frame_id)
        elif(self.frame_stage == STAGE_SECOND_FRAME):
            self.processSecondFrame()
        elif(self.frame_stage == STAGE_FIRST_FRAME):
            self.processFirstFrame()
        self.last_frame = self.new_frame