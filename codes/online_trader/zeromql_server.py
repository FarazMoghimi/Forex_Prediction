import zmq
import time

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://127.0.0.1:9999")

print("Server Started")
while True:
    time.sleep(4)
    socket.send_string("World")
    message = socket.recv()  # Wait for next request from client
    time.sleep(1)
    socket.send_string("World")