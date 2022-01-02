import socket
import cv2
import numpy


def recvall(sock, count):
    buf = b''
    while count:
        (newbuf,addr) = sock.recvfrom(count)
        print(newbuf)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

class ImageTransferClient:
    def __init__(self, host, port, server_host):
        self.host = host
        self.port = port
        self.server_host = server_host
        self.server_port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.host, self.port))

    def sendall(self, frame):
        encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
        result, imgencode = cv2.imencode('.jpg', frame, encode_param)
        data = numpy.array(imgencode)
        stringData = data.tostring()

        packet_size = 4649
        n_packets = 25

        for i in range(n_packets):
            start = i * packet_size
            end = (i+1) * packet_size
            self.sock.sendto( stringData[start:end], (self.server_host, self.server_port) )
    
    def quit(self):
        self.sock.close()

class ImageTransferServer:
    def __init__(self, port=5000):
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((socket.gethostbyname( '0.0.0.0' ), self.port))

    def recvall(self):
        packet_size = 4649
        n_packets = 25

        buf = b''
        for i in range(n_packets):
            stringData = recvall(self.sock, packet_size)

            if stringData is None:
                break

            buf += stringData
        
        data = numpy.fromstring(buf, dtype='uint8')

        if data.size == 0:
            return None

        decimg=cv2.imdecode(data,1)
        
        return decimg

    def quit(self):
        self.sock.close()
