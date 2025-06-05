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
        """
        unity_sender: an instance of UnitySender (already connected)
        host/port: where Plux‐Bitalino is streaming (default localhost:5555)
        buffer_size: how many bytes to recv() at once
        threshold: |EMG| must exceed this (in mV) to count as a contraction
        device_mac: MAC of Bitalino device
        channel_index: index within each JSON frame array where A1 lives (5)
        """
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
        # All commands must be newline-terminated
        self.sock.sendall((cmd + "\n").encode("utf-8"))

    def _recv_response(self):
        """
        Block until we get at least one complete JSON reply to a command.
        We assume command replies are small enough that a single recv() will get the full JSON.
        """
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
        """
        Main loop: connect to Bitalino, enable & start, then stream JSON frames.
        Whenever |EMG| > threshold, forward a JSON packet to Unity and print a log line.
        """
        self.connect()
        time.sleep(0.1)
        self.enable_and_start()

        print("Listening for EMG contractions (Ctrl+C to stop)...")
        try:
            while True:
                chunk = self.sock.recv(self.buffer_size)
                if not chunk:
                    print("Connection closed by Bitalino server.")
                    break

                self._recv_buffer += chunk.decode("utf-8")

                # Extract as many complete JSON objects as possible
                while True:
                    try:
                        obj, idx = self.decoder.raw_decode(self._recv_buffer)
                    except ValueError:
                        # Not enough data yet to decode a full JSON object
                        break

                    # Consume the parsed JSON portion
                    self._recv_buffer = self._recv_buffer[idx:]

                    # Each JSON object has a "returnData" field
                    rd = obj.get("returnData", {})
                    frames = rd.get(self.device_mac, [])

                    for frame in frames:
                        emg_value = frame[self.channel_index]
                        if abs(emg_value) > self.threshold:
                            timestamp = time.time()
                            print(f"[{timestamp:.3f}] Contraction detected: EMG={emg_value:.3f} mV")

                            # Send to Unity: here "emg" is the inputName; you can change it
                            self.unity_sender.send_emg("emg", emg_value)

        except KeyboardInterrupt:
            print("\nInterrupted by user. Cleaning up...")
        finally:
            # Send "stop" and close the socket
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
    # 1) First, connect to Unity’s listener at localhost:6000
    sender = UnitySender(host="127.0.0.1", port=6000)
    try:
        sender.connect()
        print("Connected to Unity on port 6000.")
    except ConnectionRefusedError:
        print("ERROR: Could not connect to Unity on port 6000. Is Unity running with EMGReceiver?")
        exit(1)

    # 2) Now pass that UnitySender into our EMG detector
    detector = SimpleEMGDetector(unity_sender=sender, threshold=0.25)
    detector.run()
