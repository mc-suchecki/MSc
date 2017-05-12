import io
import zmq
import random

from PIL import Image

# the server will listen for requests on this port
port = "6666"

# create context and specify the port
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:{}".format(port))

# wait for requests and respond
while True:
    data = socket.recv()
    print("Received a photo!")
    stream = io.BytesIO(data)
    image = Image.open(stream)
    print("Photo resolution is {}x{} pixels.".format(image.size[0], image.size[1]))
    image.show()
    score = random.randint(1, 100)
    socket.send_json({"score": score})
