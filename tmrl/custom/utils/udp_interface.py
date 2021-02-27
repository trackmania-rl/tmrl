'''
Based on https://github.com/ricardodeazambuja/BrianConnectUDP/blob/master/brian_multiprocess_udp.py
'''
import socket
import struct


class UDPInterface(object):
    def __init__(self):
        self._sockO = None
        self._sockI = None

    def __del__(self):
        if self._sockI:
            self._sockI.close()
        if self._sockO:
            self._sockO.close()

    def init_sender(self, ip, port):
        self._sockO = socket.socket(
            socket.AF_INET,  # IP
            socket.SOCK_DGRAM)  # UDP
        self._IPO = ip
        self._PORTO = port

    def init_receiver(self, ip, port, clean=True):
        self._sockI = socket.socket(
            socket.AF_INET,  # IP
            socket.SOCK_DGRAM)  # UDP

        self._sockI.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Tells the OS that if someone else is using the PORT, it
        # can use the same PORT without any error/warning msg.
        # Actually this is useful because if you restart the simulation
        # the OS is not going to release the socket so fast and an error
        # could occur.

        self._sockI.bind((ip, port))  # Bind the socket to the ip/port

        self._buffersize = self._sockI.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)

        while clean:
            print("Cleaning receiving buffer...")
            try:
                # buffer size is 1 byte, NON blocking.
                self._sockI.recv(1, socket.MSG_DONTWAIT)
            except IOError:  # The try and except are necessary because the recv raises a error when no data is received
                clean = False
        print("Cleaning receiving buffer...Done!")

        # Tells the system that the socket recv() method WILL block until a packet is received
        # self._sockI.setblocking(1)

    def send_msg(self, data):
        '''
        data: list/tuple of floats
        It will break the system if you try to send something too big...
        '''
        assert self._sockO, 'init_sender was not initialized!'

        data_header = struct.pack('>I', len(data))  # d=>8bytes
        data = data_header + \
            b''.join([struct.pack(">d", float(ji)) for ji in data])

        self._sockO.sendto(data, (self._IPO, self._PORTO))

    def recv_msg(self, timeout=None):
        """
        This returns a list of tuples (all messages in the buffer)
        Returns None if no message is present in the buffer
        Reads self._buffersize bytes in the buffer
        """
        assert self._sockI, 'init_receiver was not initialized!'
        if timeout:
            self._sockI.settimeout(timeout)
        try:
            data = self._sockI.recv(self._buffersize)
        except socket.timeout:
            return None
        s = 0
        i = 4
        ld = len(data)
        res = []
        while i < ld:
            # print(f"DEBUG: s,i={s},{i}")
            msglen = struct.unpack('>I', data[s:i])[0]
            res.append(struct.unpack('>' + ''.join(['d'] * msglen), data[i:i + 8 * msglen]))
            # print(f"DEBUG: msglen={msglen}, res={res}")
            s = i + 8 * msglen
            i = s + 4
        nb = self.recv_msg_nonblocking()
        if nb:
            res = res + nb
        return res

    def recv_msg_nonblocking(self):
        """
        This returns a list of tuples (all messages in the buffer)
        Returns None if no message is present in the buffer
        Reads everything in the buffer
        """
        assert self._sockI, 'init_receiver was not initialized!'
        data = b''
        while True:
            try:
                packet = self._sockI.recv(self._buffersize, socket.MSG_DONTWAIT)
            except IOError:
                s = 0
                i = 4
                if data:
                    ld = len(data)
                    res = []
                    while i < ld:
                        # print(f"DEBUG: s,i={s},{i}")
                        msglen = struct.unpack('>I', data[s:i])[0]
                        res.append(struct.unpack('>' + ''.join(['d'] * msglen), data[i:i + 8 * msglen]))
                        # print(f"DEBUG: msglen={msglen}, res={res}")
                        s = i + 8 * msglen
                        i = s + 4
                    return res
                else:
                    return None
            data += packet


def main(args):
    import time

    ip_send = args.ipsend
    ip_recv = args.iprecv
    port_send = args.portsend
    port_recv = args.portrecv
    throttle = args.throttle
    justlistenforever = args.justlistenforever
    conn = UDPInterface()
    conn.init_sender(ip_send, port_send)
    conn.init_receiver(ip_recv, port_recv)
    if justlistenforever:
        tick_1 = time.time()
        while True:
            res = conn.recv_msg()
            tick_2 = time.time()
            print("---")
            print(f"received (blocking): {res}")
            print(f"elapsed time since last received: {tick_2 - tick_1} s")
            tick_1 = tick_2
    else:
        # [roll, pitch, yaw, throttle, arm/disarm]
        msg_sent = [1500, 1500, 1500, throttle, 1.0]
        print(f"sending: {msg_sent}")
        conn.send_msg(msg_sent)
        print(f"sending: {msg_sent}")
        conn.send_msg(msg_sent)
        tick_1 = time.time()
        res = conn.recv_msg()
        tick_2 = time.time()
        print(f"received (blocking): {res}")
        print(f"elapsed time between sent and received: {tick_2 - tick_1} s")
        print(f"sending: {msg_sent}")
        conn.send_msg(msg_sent)
        print("sleeping...")
        time.sleep(0.1)
        res = conn.recv_msg_nonblocking()
        print(f"received (nonblocking): {res}")
        print(f"last message: {res[-1]}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ipsend', type=str, default=None, help='IP address of the drone if any.')
    parser.add_argument('--iprecv', type=str, default=None, help='local IP address if any.')
    parser.add_argument('--portsend', type=int, default=8989, help='Port to send udp messages to.')
    parser.add_argument('--portrecv', type=int, default=8989, help='Port to reveive udp messages from.')
    parser.add_argument('--throttle', type=int, default=1650, help='Throttle action to send.')
    parser.add_argument('--justlistenforever', type=bool, default=False, help='Infinite recv client')
    args = parser.parse_args()
    main(args)

# if __name__ == '__main__':
#     import time
#     conn = UDPInterface()
#
#     conn.init_sender('127.0.0.1', 8989)
#     conn.init_receiver('127.0.0.1', 8989)
#
#     conn.send_msg([1, 2, 3])
#     conn.send_msg([4, 5, 6])
#     time.sleep(0.1)
#     print(conn.recv_msg())
#
#     conn.send_msg([1, 2, 3])
#     conn.send_msg([4, 5, 6])
#     time.sleep(0.1)
#     res = conn.recv_msg_nonblocking()
#     print(res)
#     print(f"last message: {res[-1]}")
