from agents.envs import UntouchedGymEnv
from agents.util import load
from agents.sac_models import *
from threading import Lock, Thread
from agents.util import collate, partition, partial

from collections import deque
import gym
from copy import deepcopy
import socket
import select
from requests import get
import pickle
from argparse import ArgumentParser
import time


PORT_TRAINER = 55555  # Port to listen on (non-privileged ports are > 1023)
PORT_ROLLOUT = 55556  # Port to listen on (non-privileged ports are > 1023)
BUFFER_SIZE = 1024  # socket buffer
HEADER_SIZE = 12  # fixed number of characters used to describe the data length

SOCKET_TIMEOUT_CONNECT_TRAINER = 30.0
SOCKET_TIMEOUT_ACCEPT_TRAINER = 30.0
SOCKET_TIMEOUT_CONNECT_ROLLOUT = 30.0
SOCKET_TIMEOUT_ACCEPT_ROLLOUT = 30.0  # socket waiting for rollout workers closed and restarted at this interval

SELECT_TIMEOUT_INBOUND_REDIS_FROM_TRAINER = 30.0  #  redis <- trainer (weights)
SELECT_TIMEOUT_INBOUND_TRAINER_FROM_REDIS = 30.0  #  trainer <- redis (full samples batch)

SELECT_TIMEOUT_INBOUND_ROLLOUT = 30.0

SOCKET_TIMEOUT_COMMUNICATE = 30.0
SELECT_TIMEOUT_OUTBOUND = 30.0

SELECT_TIMEOUT_PING_PONG = 60.0

WAIT_BEFORE_RECONNECTION = 10.0

def ping_pong(sock):
    """
    This pings and waits for pong
    All inbound select() calls must expect to receive a PING request and answer with PONG
    returns True if success, False otherwise
    """
    _, wl, xl = select.select([], [sock], [sock], SELECT_TIMEOUT_OUTBOUND)  # select for writing
    if len(xl) != 0 or len(wl) == 0:
        print("INFO: socket error/timeout while sending PING")
        return False
    send_ping(sock)
    rl, _, xl = select.select([sock], [], [sock], SELECT_TIMEOUT_PING_PONG)  # select for reading
    if len(xl) != 0 or len(rl) == 0:
        print("INFO: socket error/timeout while waiting for PONG")
        return False
    obj = recv_object(sock)
    if obj == 'PONG':
        return True
    else:
        return False


def send_ping(sock):
    return send_object(sock, None, ping=True, pong=False)


def send_pong(sock):
    return send_object(sock, None, ping=False, pong=True)


def send_object(sock, obj, ping=False, pong=False):
    """
    If ping, this will ignore obj and send the PING request
    If pong, this will ignore obj and send the PONG request
    If raw, obj must be a binary string
    Call only after select on a socket with a (long enough) timeout.
    Returns True if sent successfully, False if connection lost.
    """
    if ping:
        msg = bytes(f"{'PING':<{HEADER_SIZE}}", 'utf-8')
    elif pong:
        msg = bytes(f"{'PONG':<{HEADER_SIZE}}", 'utf-8')
    else:
        msg = pickle.dumps(obj)
        msg = bytes(f"{len(msg):<{HEADER_SIZE}}", 'utf-8') + msg
    try:
        sock.sendall(msg)
    except OSError:  # connection closed or broken
        return False
    return True


def recv_object(sock):
    """
    If the request is PING or PONG, this will return 'PINGPONG'
    If the request is PING, this will automatically send the PONG answer
    Call only after select on a socket with a (long enough) timeout.
    Returns the object if received successfully, None if connection lost.
    """
    # first, we receive the header (inefficient but prevents collisions)
    msg = b''
    l = len(msg)
    while l != HEADER_SIZE:
        try:
            recv_msg = sock.recv(HEADER_SIZE - l)
            if len(recv_msg) == 0:  # connection closed or broken
                return None
            msg += recv_msg
        except OSError:  # connection closed or broken
            return None
        l = len(msg)
        # print(f"DEBUG1: l:{l}")
    # print("DEBUG: data len:", msg[:HEADER_SIZE])
    print(f"DEBUG: msg[:4]: {msg[:4]}")
    if msg[:4] == 'PING' or msg[:4] == 'PONG':
        if msg[:4] == 'PING':
            send_pong(sock)
        return 'PINGPONG'
    msglen = int(msg[:HEADER_SIZE])
    # now, we receive the actual data (no more than the data length, again to prevent collisions)
    msg = b''
    l = len(msg)
    while l != msglen:
        try:
            recv_msg = sock.recv(min(BUFFER_SIZE, msglen - l))  # this will not receive more bytes than required
            if len(recv_msg) == 0:  # connection closed or broken
                return None
            msg += recv_msg
        except OSError:  # connection closed or broken
            return None
        l = len(msg)
        # print(f"DEBUG2: l:{l}")
    # print("DEBUG: final data len:", l)
    return pickle.loads(msg)


def get_listening_socket(timeout, ip_bind, port_bind):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.bind((ip_bind, port_bind))
    s.listen(5)
    return s


def get_connected_socket(timeout, ip_connect, port_connect):
    """
    returns the connected socket
    returns None if connect failed
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect((ip_connect, port_connect))
    except OSError:  # connection broken or timeout
        print("INFO: connect() timed-out or failed")
        s.close()
        print(f"INFO: sleeping {WAIT_BEFORE_RECONNECTION}s")
        time.sleep(WAIT_BEFORE_RECONNECTION)
        return None
    s.settimeout(SOCKET_TIMEOUT_COMMUNICATE)
    return s


def accept_or_close_socket(s):
    """
    returns conn, addr
    None None in case of failure
    """
    conn = None
    try:
        conn, addr = s.accept()
        conn.settimeout(SOCKET_TIMEOUT_COMMUNICATE)
        return conn, addr
    except OSError:
        print("INFO: accept() timed-out or failed")
        if conn is not None:
            conn.close()
        s.close()
        print(f"INFO: sleeping {WAIT_BEFORE_RECONNECTION}s")
        time.sleep(WAIT_BEFORE_RECONNECTION)
        return None, None


def select_and_send_or_close_socket(obj, conn):
    """
    Returns True if success
    False if disconnected (closes sockets)
    """
    _, wl, xl = select.select([], [conn], [conn], SELECT_TIMEOUT_OUTBOUND)  # select for writing
    if len(xl) != 0:
        print("INFO: error when writing, closing sockets")
        conn.close()
        return False
    if len(wl) == 0:
        print("INFO: select timed out, sending PING request")
        if not ping_pong(conn):
            print("INFO: ping pong failed, closing sockets")
            conn.close()
            return False
    elif not send_object(conn, obj):  # error or timeout
        print("INFO: send_object() failed, closing sockets")
        conn.close()
        return False
    return True


def poll_and_recv_or_close_socket(conn):
    """
    Returns True, obj is success (obj is None if nothing was in the read buffer when polling)
    False, None otherwise
    """
    rl, _, xl = select.select([conn], [], [conn], 0.0)  # polling read channel
    if len(xl) != 0:
        print("INFO: error when polling, closing sockets")
        conn.close()
        return False, None
    if len(rl) == 0:  # nothing in the recv buffer
        return True, None
    obj = recv_object(conn)
    if obj is None:  # socket error
        print("INFO: error when receiving object, closing sockets")
        conn.close()
        return False, None
    elif obj == 'PINGPONG':
        return True, None
    else:
        print(f"DEBUG: received obj:{obj}")
        return True, obj


# BUFFER: ===========================================

class Buffer:
    def __init__(self):
        self.memory = []

    def append_sample(self, obs, rew, done, info):
        self.memory.append((obs, rew, done, info, ))

    def clear(self):
        self.memory = []


# REDIS SERVER: =====================================

class RedisServer:
    """
    This is the main server
    This lets 1 TrainerInterface and n RolloutWorkers connect
    This buffers experiences sent by RolloutWorkers
    This periodically sends the buffer to the TrainerInterface
    This also receives the weights from the TrainerInterface and broadcast them to the connected RolloutWorkers
    """
    def __init__(self, samples_per_redis_batch=1000):
        self.__buffer = Buffer()
        self.__buffer_lock = Lock()
        self.__weights_lock = Lock()
        self.__weights = None
        self.samples_per_redis_batch = samples_per_redis_batch
        self.public_ip = get('http://api.ipify.org').text
        self.local_ip = socket.gethostbyname(socket.gethostname())

        print(f"INFO REDIS: local IP: {self.local_ip}")
        print(f"INFO REDIS: public IP: {self.public_ip}")

        Thread(target=self.__rollout_workers_thread, args=(), kwargs={}, daemon=True).start()
        Thread(target=self.__trainer_thread, args=(), kwargs={}, daemon=True).start()

        print("DEBUG: REACH THIS POINT")

    def __trainer_thread(self, ):
        """
        This waits for a TrainerInterface to connect
        Then, this periodically sends the local buffer to the TrainerInterface (when data is available)
        When the TrainerInterface sends new weights, this broadcasts them to all connected RolloutWorkers
        """
        while True:  # main redis loop
            s = get_listening_socket(SOCKET_TIMEOUT_ACCEPT_TRAINER, self.local_ip, PORT_TRAINER)
            conn, addr = accept_or_close_socket(s)
            if conn is None:
                print("DEBUG: accept_or_close_socket failed in trainer thread")
                continue
            print(f"INFO TRAINER THREAD: redis connected by trainer at address {addr}")
            # Here we could spawn a Trainer communication thread, but since there is only one trainer we move on
            i = 0
            while True:
                # send samples
                self.__buffer_lock.acquire()  # BUFFER LOCK.............................................................
                if len(self.__buffer) >= self.samples_per_redis_batch:
                    obj = self.__buffer
                    print(f"INFO TRAINER THREAD: sending obj {i}")
                    if not select_and_send_or_close_socket(obj, conn):
                        self.__buffer_lock.release()
                        break
                    self.__buffer.clear()
                self.__buffer_lock.release()  # END BUFFER LOCK.........................................................
                # checks for weights
                success, obj = poll_and_recv_or_close_socket(conn)
                if not success:
                    print("DEBUG: poll failed in trainer thread")
                    break
                elif obj is not None:
                    print(f"DEBUG INFO: trainer thread received obj")
                    self.__weights_lock.acquire()  # WEIGHTS LOCK.......................................................
                    self.__weights = obj
                    self.__weights_lock.release()  # END WEIGHTS LOCK...................................................
                time.sleep(10.0)  # TODO: adapt
                i += 1
            s.close()

    def __rollout_workers_thread(self):
        """
        This waits for new potential RolloutWorkers to connect
        When a new RolloutWorker connects, this instantiates a new thread to handle it
        """
        while True:  # main redis loop
            s = get_listening_socket(SOCKET_TIMEOUT_ACCEPT_ROLLOUT, self.local_ip, PORT_ROLLOUT)
            conn, addr = accept_or_close_socket(s)
            if conn is None:
                print("DEBUG: accept_or_close_socket failed in workers thread")
                continue
            print(f"INFO WORKERS THREAD: redis connected by worker at address {addr}")
            Thread(target=self.__rollout_worker_thread, args=(conn, ), kwargs={}, daemon=True).start()  # we don't keep track of this for now
            s.close()

    def __rollout_worker_thread(self, conn):
        """
        Thread handling connection to a single RolloutWorker
        """
        while True:
            # send weights
            self.__weights_lock.acquire()  # WEIGHTS LOCK...............................................................
            if self.__weights is not None:  # new weigths
                obj = self.__weights
                if not select_and_send_or_close_socket(obj, conn):
                    self.__weights_lock.release()
                    print("DEBUG: select_and_send_or_close_socket failed in worker thread")
                    break
                self.__weights = None
            self.__weights_lock.release()  # END WEIGHTS LOCK...........................................................
            # checks for samples
            success, obj = poll_and_recv_or_close_socket(conn)
            if not success:
                print("DEBUG: poll failed in rollout thread")
                break
            elif obj is not None:
                print(f"DEBUG INFO: rollout worker thread received obj")
                self.__buffer_lock.acquire()  # BUFFER LOCK.............................................................
                self.__buffer += obj  # concat worker batch to local batch
                self.__buffer_lock.release()  # END BUFFER LOCK.........................................................
            time.sleep(10.0)  # TODO: adapt


# TRAINER: ==========================================

class TrainerInterface:
    """
    This is the trainer's network interface
    This connects to the redis server
    This receives samples batches and send new weights
    """
    def __init__(self,
                 ip_redis=None,
                 model_path=r'C:/Users/Yann/Desktop/git/tmrl/checkpoint/weights/expt.pth'):
        self.__buffer_lock = Lock()
        self.__weights_lock = Lock()
        self.__weights = None
        self.__buffer = Buffer()
        self.model_path = model_path
        self.public_ip = get('http://api.ipify.org').text
        self.local_ip = socket.gethostbyname(socket.gethostname())

        print(f"local IP: {self.local_ip}")
        print(f"public IP: {self.public_ip}")

        self.ip_redis = ip_redis if ip_redis is not None else self.local_ip

        Thread(target=self.__run_thread, args=(), kwargs={}, daemon=True).start()

    def __run_thread(self):
        """
        Trainer interface thread
        """
        while True:  # main client loop
            s = get_connected_socket(SOCKET_TIMEOUT_CONNECT_TRAINER, self.public_ip, PORT_TRAINER)
            if s is None:
                print("DEBUG: get_connected_socket failed in TrainerInterface thread")
                continue
            while True:
                # send weights
                self.__weights_lock.acquire()  # WEIGHTS LOCK...........................................................
                if self.__weights is not None:  # new weights
                    obj = self.__weights
                    if not select_and_send_or_close_socket(obj, s):
                        self.__weights_lock.release()
                        print("DEBUG: select_and_send_or_close_socket failed in trainerinterface")
                        break
                    self.__weights = None
                self.__weights_lock.release()  # END WEIGHTS LOCK.......................................................
                # checks for samples batch
                success, obj = poll_and_recv_or_close_socket(s)
                if not success:
                    print("DEBUG: poll failed in TrainerInterface thread")
                    break
                elif obj is not None:  # received buffer
                    print(f"DEBUG INFO: trainer interface received obj")
                    self.__buffer_lock.acquire()  # BUFFER LOCK.....................................................................
                    self.__buffer += obj
                    self.__buffer_lock.release()  # END BUFFER LOCK.................................................................
                time.sleep(10.0)  # TODO: adapt
            s.close()

    def broadcast_model(self, model: ActorModule):
        """
        model must be an ActorModule (sac_models.py)
        broadcasts the model's weights to all connected RolloutWorkers
        """
        self.__weights_lock.acquire()  # WEIGHTS LOCK...................................................................
        torch.save(model.state_dict(), self.model_path)
        with open(self.model_path, 'rb') as f:
            self.__weights = f.read()
        self.__weights_lock.release()  # END WEIGHTS LOCK...............................................................

    def retrieve_buffer(self):
        """
        updates the Trainer's replay buffer with the TrainerInterface's local buffer
        empties the local buffer
        """
        self.__buffer_lock.acquire()  # BUFFER LOCK.....................................................................
        if len(self.__buffer) > 0:
            pass  # TODO: update trainer's replay buffer
            self.__buffer.clear()
        self.__buffer_lock.release()  # END BUFFER LOCK.................................................................


# ROLLOUT WORKER: ===================================

class RolloutWorker:
    def __init__(self,
                 env_id,
                 actor_module_cls,
                 # obs_space,
                 # act_space,
                 device="cpu",
                 ip_redis=None,
                 samples_per_worker_batch=1000,
                 sleep_between_batches=0.0,
                 model_path=r"C:/Users/Yann/Desktop/git/tmrl/checkpoint/weights/exp.pth"
                 ):
        self.env = UntouchedGymEnv(id=env_id)
        obs_space = self.env.observation_space
        act_space = self.env.action_space
        self.model_path = model_path
        self.actor = actor_module_cls(obs_space, act_space)
        self.actor.load_state_dict(torch.load(self.model_path))
        self.device = device
        self.buffer = Buffer()
        self.__buffer = Buffer()  # deepcopy for sending
        self.__buffer_lock = Lock()
        self.__weights = None
        self.__weights_lock = Lock()
        self.samples_per_worker_batch = samples_per_worker_batch
        self.sleep_between_batches = sleep_between_batches

        self.public_ip = get('http://api.ipify.org').text
        self.local_ip = socket.gethostbyname(socket.gethostname())

        print(f"local IP: {self.local_ip}")
        print(f"public IP: {self.public_ip}")

        self.ip_redis = ip_redis if ip_redis is not None else self.public_ip

        Thread(target=self.__run_thread, args=(), kwargs={}, daemon=True).start()

    def __run_thread(self):
        """
        Trainer interface thread
        """
        while True:  # main client loop
            s = get_connected_socket(SOCKET_TIMEOUT_CONNECT_ROLLOUT, self.ip_redis, PORT_ROLLOUT)
            if s is None:
                print("DEBUG: get_connected_socket failed in worker")
                continue
            while True:
                # send buffer
                self.__buffer_lock.acquire()  # BUFFER LOCK.............................................................
                if len(self.__buffer) >= self.samples_per_worker_batch:  # a new batch is available
                    obj = self.__buffer
                    if not select_and_send_or_close_socket(obj, s):
                        self.__buffer_lock.release()
                        print("DEBUG: select_and_send_or_close_socket failed in worker")
                        break
                self.__buffer.clear()  # empty sent batch
                self.__buffer_lock.release()  # END BUFFER LOCK.........................................................
                # checks for new weights
                success, obj = poll_and_recv_or_close_socket(s)
                if not success:
                    print(f"INFO: rollout worker poll failed")
                    break
                elif obj is not None:
                    print(f"DEBUG INFO: rollout worker received obj")
                    self.__weights_lock.acquire()  # WEIGHTS LOCK.......................................................
                    self.__weights = obj
                    self.__weights_lock.release()  # END WEIGHTS LOCK...................................................
                time.sleep(10.0)  # TODO: adapt
            s.close()

    def act(self, obs, train=False):
        """
        converts inputs to torch tensors and converts outputs to numpy arrays
        """
        obs = collate([obs], device=self.device)
        with torch.no_grad():
            action_distribution = self.actor(obs)
            action = action_distribution.sample() if train else action_distribution.sample_deterministic()
        action, = partition(action)
        return action

    def collect_n_steps(self, n, train=True):
        """
        collects n transitions
        set train to False for test samples, True for train samples
        """
        # self.buffer.clear()
        obs = self.env.reset()
        print(f"DEBUG: init obs[0]:{obs[0]}")
        print(f"DEBUG: init obs[1][-1].shape:{obs[1][-1].shape}")
        obs_mod = (obs[0], obs[1][-1], )  # speed and most recent image
        self.buffer.append_sample(obs_mod, 0.0, False, {})
        for _ in range(n):
            act = self.act(obs, train)
            obs, rew, done, info = self.env.step(act)
            obs_mod = (obs[0], obs[1][-1],)  # speed and most recent image only
            self.buffer.append_sample(obs_mod, rew, done, info)

    def send_and_clear_buffer(self):
        self.__buffer_lock.acquire()  # BUFFER LOCK.....................................................................
        self.__buffer = deepcopy(self.buffer)
        self.__buffer_lock.release()  # END BUFFER LOCK.................................................................
        self.buffer.clear()

    def update_actor_weights(self):
        """
        updates the model with new weights from the trainer when available
        """
        self.__weights_lock.acquire()  # WEIGHTS LOCK...................................................................
        if self.__weights is not None:  # new weights available
            with open(self.model_path, 'wb') as f:  # FIXME: check that this deletes the old file
                f.write(self.__weights)
            self.actor.load_state_dict(torch.load(self.model_path))
            self.__weights = None
        self.__weights_lock.release()  # END WEIGHTS LOCK...............................................................



def main_old(args):
    redis = args.redis
    public_ip = get('http://api.ipify.org').text
    local_ip = socket.gethostbyname(socket.gethostname())
    print(f"I: local IP: {local_ip}")
    print(f"I: public IP: {public_ip}")

    if redis:  # server (redis)
        while True:  # main redis loop
            # TODO: we can close and restart s without closing conn
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(SOCKET_TIMEOUT_ACCEPT_ROLLOUT)
            s.bind((local_ip, PORT_ROLLOUT))
            s.listen(5)
            try:
                conn, addr = s.accept()
                conn.settimeout(SOCKET_TIMEOUT_COMMUNICATE)
            except OSError:
                print("INFO: accept() timed-out or failed")
                conn.close()
                # s.close()
                print("INFO: sleeping 10 sec...")
                time.sleep((10.0))
                continue
            # TODO: launch a new thread here
            print(f"INFO: redis connected by address {addr}")
            i = 0
            while True:
                obj = {"i": i, "time": time.time()}
                print(f"REDIS: sending obj {i}")
                _, wl, xl = select.select([], [conn], [conn], SELECT_TIMEOUT_OUTBOUND)  # select for writing
                if len(xl) != 0 or len(wl) == 0 or not send_object(wl[0], obj):  # error or timeout
                    print("INFO: select outbound/send_object() timed-out or failed")
                    conn.close()
                    break
                time.sleep(10.0)
                i += 1
            s.close()
    else:  # client (rollout worker)s
        while True:  # main client loop
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(SOCKET_TIMEOUT_CONNECT_ROLLOUT)
            try:
                s.connect((public_ip, PORT_ROLLOUT))
            except OSError:  # connection broken or timeout
                print("INFO: connect() timed-out or failed")
                s.close()
                print("INFO: sleeping 10 sec...")
                time.sleep((10.0))
                continue
            s.settimeout(SOCKET_TIMEOUT_COMMUNICATE)
            while True:
                rl, wl, xl = select.select([s], [], [s], SELECT_TIMEOUT_INBOUND_ROLLOUT)  # select for writing
                if len(rl) == 0 or len(xl) != 0:
                    print("INFO: inbound select() timed-out or failed")
                    break
                obj = recv_object(s)
                if obj is None:  # connection broken or timeout
                    print("INFO: recv_object() timed-out or failed")
                    break
                print(f"CLIENT: received object {obj['i']} with latency: {time.time() - obj['time']}")
            s.close()


def main(args):
    redis = args.redis
    trainer = args.trainer

    public_ip = get('http://api.ipify.org').text
    local_ip = socket.gethostbyname(socket.gethostname())
    print(f"I: local IP: {local_ip}")
    print(f"I: public IP: {public_ip}")

    if redis:
        rs = RedisServer(samples_per_redis_batch=100)
    elif trainer:
        ti = TrainerInterface(ip_redis=public_ip,
                              model_path=r"C:/Users/Yann/Desktop/git/tmrl/checkpoint/weights/expt.pth")
    else:
        rw = RolloutWorker(env_id="gym_tmrl:gym-tmrl-v0",
                           actor_module_cls=partial(TMPolicy, act_in_obs=True),
                           device="cpu",
                           ip_redis=public_ip,
                           samples_per_worker_batch=100,
                           sleep_between_batches=0.0,
                           model_path=r"C:/Users/Yann/Desktop/git/tmrl/checkpoint/weights/exp.pth")
        rw.collect_n_steps(100, True)
        rw.send_and_clear_buffer()
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--redis', dest='redis', action='store_true')
    parser.set_defaults(redis=False)
    parser.add_argument('--trainer', dest='trainer', action='store_true')
    parser.set_defaults(trainer=False)
    parser.add_argument('--worker', dest='worker', action='store_true')  # not used
    parser.set_defaults(worker=False)
    args = parser.parse_args()
    main(args)
