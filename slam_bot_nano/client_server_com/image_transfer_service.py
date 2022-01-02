import socket
import cv2
import numpy


def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

class ImageTransferClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sock = socket.socket()
        self.sock.connect((self.host, self.port))

    def sendall(self, frame):
        encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
        result, imgencode = cv2.imencode('.jpg', frame, encode_param)
        data = numpy.array(imgencode)
        stringData = data.tostring()

        self.sock.send( str(len(stringData)).ljust(16))
        self.sock.send( stringData )
    
    def quit(self):
        self.sock.close()

class ImageTransferServer:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((TCP_IP, TCP_PORT))
        self.sock.listen(True)

    def recvall():
        self.conn, self.addr = self.sock.accept()
        length = recvall(self.conn,16)
        stringData = recvall(self.conn, int(length))
        data = numpy.fromstring(stringData, dtype='uint8')
        decimg=cv2.imdecode(data,1)
        self.conn.close()

        return decimg

    def quit(self):
        self.sock.close()
