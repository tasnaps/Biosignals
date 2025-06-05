import socket
import json
import time


class UnitySender:
    def __init__(self, host="127.0.0.1", port=6000):
        self.host = host
        self.port = port
        self.sock = None
        self.file = None

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        self.file = self.sock.makefile("wb")

    def send_emg(self, inputName: str, value: float):
        packet = {
            "timestamp": time.time(),
            "input": inputName,
            "value": value,
        }
        raw = (json.dumps(packet) + "\n").encode("utf-8")
        try:
            self.sock.sendall(raw)
        except BrokenPipeError:
            print("UnitySender: Connection lost")

    def close(self):
        if self.sock:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            self.sock.close()
            self.sock = None
