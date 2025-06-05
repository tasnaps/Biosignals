import socket
import json
import time


class UnitySender:
    def __init__(self, host="127.0.0.1", port=6000):
        """
        A small TCP client that connects to a Unity listener on localhost:6000 (by default).
        When send_emg() is called, it JSON‐encodes {timestamp, input, value} and appends '\n'.
        """
        self.host = host
        self.port = port
        self.sock = None
        self.file = None

    def connect(self):
        """
        Establish a TCP connection to Unity’s listener.
        If Unity is not running or not listening yet, this will block or raise ConnectionRefusedError.
        """
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        # Wrap the socket in a file-like object for potential extensions (not strictly needed here)
        self.file = self.sock.makefile("wb")

    def send_emg(self, inputName: str, value: float):
        """
        Send a JSON‐serialized EMG packet, newline‐terminated.
        Example: {"timestamp": 1717571234.56, "input": "emg", "value": 0.72}
        """
        packet = {
            "timestamp": time.time(),
            "input": inputName,
            "value": value,
        }
        raw = (json.dumps(packet) + "\n").encode("utf-8")
        try:
            self.sock.sendall(raw)
        except BrokenPipeError:
            print("UnitySender: Connection lost (BrokenPipe).")

    def close(self):
        """
        Shutdown and close the socket.
        """
        if self.sock:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            self.sock.close()
            self.sock = None
