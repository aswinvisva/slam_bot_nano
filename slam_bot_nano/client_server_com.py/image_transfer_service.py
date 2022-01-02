import redis
import cv2
import numpy as np

class ImageTransferService:
    def __init__(self, host='localhost', port=6379):
        self.port = port
        self.host = host
        self.conn = redis.Redis(host,port)
        self.frameNum = 0

    def ping(self):
        return self.conn.ping()

    def sendImage(self,im, name='latest', Q=75):
        _, JPEG = cv2.imencode(".JPG", im, [int(cv2.IMWRITE_JPEG_QUALITY), Q])
        myDict = { 'frameNum': self.frameNum, 'Data':JPEG.tobytes() }
        self.conn.hmset(name, myDict)
        self.frameNum += 1

    def receiveImage(self,name='latest'):
        myDict  = self.conn.hgetall(name)
        Data = myDict.get(b'Data')
        im = cv2.imdecode(np.frombuffer(Data,dtype=np.uint8), cv2.IMREAD_COLOR)
        return im
