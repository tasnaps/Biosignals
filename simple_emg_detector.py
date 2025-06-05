from emg_to_unity import UnitySender
import socket
import json
import time


class SimpleEMGDetector:
    def __init__(
        self,
        unity_sender: UnitySender,
        host: str = "127.0.0.1",
        port: int = 5555,
        buffer_size: int = 4096,
        threshold: float = 0.05,
        device_mac: str = "98:D3:41:FE:2E:A0",
        channel_index: int = 5,
    ):

        self.unity_sender = unity_sender
        self.host = host
        self.port = port
        self.buffer_size = buffer_size
        self.threshold = threshold
        self.device_mac = device_mac
        self.channel_index = channel_index

        # set up the TCP socket to Bitalino
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.decoder = json.JSONDecoder()
        self._recv_buffer = ""

    def connect(self):
        self.sock.connect((self.host, self.port))
        print(f"Connected to Bitalino server at {self.host}:{self.port}")

    def send_command(self, cmd: str):
        self.sock.sendall((cmd + "\n").encode("utf-8"))

    def _recv_response(self):
        data = self.sock.recv(self.buffer_size).decode("utf-8")
        return json.loads(data)

    def enable_and_start(self):
        # 1) enable the device
        self.send_command(f"enable,{self.device_mac}")
        resp = self._recv_response()
        print("Enable response:", resp)

        time.sleep(0.1)

        # 2) start streaming
        self.send_command("start")
        resp = self._recv_response()
        print("Start response:", resp)

    def run(self):
        self.connect()
        time.sleep(0.1)
        self.enable_and_start()

        print("Listening for EMG contractions")
        try:
            while True:
                chunk = self.sock.recv(self.buffer_size)
                if not chunk:
                    print("Connection closed by Bitalino server.")
                    break
                self._recv_buffer += chunk.decode("utf-8")
                while True:
                    try:
                        obj, idx = self.decoder.raw_decode(self._recv_buffer)
                    except ValueError:
                        break
                    self._recv_buffer = self._recv_buffer[idx:]
                    rd = obj.get("returnData", {})
                    frames = rd.get(self.device_mac, [])
                    for frame in frames:
                        emg_value = frame[self.channel_index]
                        if abs(emg_value) > self.threshold:
                            timestamp = time.time()
                            print(f"[{timestamp:.3f}] Contraction detected: EMG={emg_value:.3f} mV")
                            self.unity_sender.send_emg("emg", emg_value)

        except KeyboardInterrupt:
            print("Keyboard interruption")
        finally:
            try:
                self.send_command("stop")
                stop_resp = self._recv_response()
                print("Stop response:", stop_resp)
            except Exception:
                pass
            self.sock.close()
            self.unity_sender.close()
            print("Sockets closed. Exiting.")


if __name__ == "__main__":
    sender = UnitySender(host="127.0.0.1", port=6000)
    try:
        sender.connect()
        print("Connected to Unity on port 6000.")
    except ConnectionRefusedError:
        print("ERROR: Could not connect to Unity on port 6000.")
        exit(1)
    # Unity sender yay
    detector = SimpleEMGDetector(unity_sender=sender, threshold=0.25)
    detector.run()
