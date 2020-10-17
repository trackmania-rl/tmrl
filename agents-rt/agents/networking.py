from copy import deepcopy
import select
import pickle
import torch
import time
from threading import Thread, Lock
import numpy as np
from requests import get
import socket
import os

from agents.sac_models import ActorModule
from agents.envs import UntouchedGymEnv
from agents.util import collate, partition
import agents.custom.config as cfg


# NETWORK: ==========================================

def ping_pong(sock):
    """
    This pings and waits for pong
    All inbound select() calls must expect to receive PINGPING
    returns True if success, False otherwise
    closes socket if failed
    # FIXME: do not use, the pingpong mechanism interferes with transfers
    """
    _, wl, xl = select.select([], [sock], [sock], cfg.SELECT_TIMEOUT_OUTBOUND)  # select for writing
    if len(xl) != 0 or len(wl) == 0:
        print("INFO: socket error/timeout while sending PING")
        sock.close()
        return False
    send_ping(sock)
    rl, _, xl = select.select([sock], [], [sock], cfg.SELECT_TIMEOUT_PING_PONG)  # select for reading
    if len(xl) != 0 or len(rl) == 0:
        print("INFO: socket error/timeout while waiting for PONG")
        sock.close()
        return False
    obj = recv_object(sock)
    if obj == 'PINGPONG':
        return True
    else:
        print("INFO: PINGPONG received an object that is not PING or PONG")
        sock.close()
        return False


def send_ping(sock):
    return send_object(sock, None, ping=True, pong=False, ack=False)


def send_pong(sock):
    return send_object(sock, None, ping=False, pong=True, ack=False)


def send_ack(sock):
    return send_object(sock, None, ping=False, pong=False, ack=True)


def send_object(sock, obj, ping=False, pong=False, ack=False):
    """
    If ping, this will ignore obj and send the PING request
    If pong, this will ignore obj and send the PONG request
    If ack, this will ignore obj and send the ACK request
    If raw, obj must be a binary string
    Call only after select on a socket with a (long enough) timeout.
    Returns True if sent successfully, False if connection lost.
    """
    if ping:
        msg = bytes(f"{'PING':<{cfg.HEADER_SIZE}}", 'utf-8')
    elif pong:
        msg = bytes(f"{'PONG':<{cfg.HEADER_SIZE}}", 'utf-8')
    elif ack:
        msg = bytes(f"{'ACK':<{cfg.HEADER_SIZE}}", 'utf-8')
    else:
        msg = pickle.dumps(obj)
        msg = bytes(f"{len(msg):<{cfg.HEADER_SIZE}}", 'utf-8') + msg
    try:
        nb_bytes = len(msg) - cfg.HEADER_SIZE
        # if nb_bytes > 0:
        #     print(f"DEBUG: sending object of {nb_bytes} bytes")
        t_start = time.time()
        sock.sendall(msg)
        # if nb_bytes > 0:
        #     print(f"DEBUG: finished sending after {time.time() - t_start}s")
    except OSError:  # connection closed or broken
        return False
    return True


def recv_object(sock):
    """
    If the request is PING or PONG, this will return 'PINGPONG'
    If the request is ACK, this will return 'ACK'
    If the request is PING, this will automatically send the PONG answer
    Call only after select on a socket with a (long enough) timeout.
    Returns the object if received successfully, None if connection lost.
    This sends the ACK request back to sock when an object transfer is complete
    """
    # first, we receive the header (inefficient but prevents collisions)
    msg = b''
    l = len(msg)
    while l != cfg.HEADER_SIZE:
        try:
            recv_msg = sock.recv(cfg.HEADER_SIZE - l)
            if len(recv_msg) == 0:  # connection closed or broken
                return None
            msg += recv_msg
        except OSError:  # connection closed or broken
            return None
        l = len(msg)
        # print(f"DEBUG: l:{l}")
    # print("DEBUG: data len:", msg[:HEADER_SIZE])
    # print(f"DEBUG: msg[:4]: {msg[:4]}")
    if msg[:4] == b'PING' or msg[:4] == b'PONG':
        if msg[:4] == b'PING':
            send_pong(sock)
        return 'PINGPONG'
    if msg[:3] == b'ACK':
        return 'ACK'
    msglen = int(msg[:cfg.HEADER_SIZE])
    # print(f"DEBUG: receiving {msglen} bytes")
    t_start = time.time()
    # now, we receive the actual data (no more than the data length, again to prevent collisions)
    msg = b''
    l = len(msg)
    while l != msglen:
        try:
            recv_msg = sock.recv(min(cfg.BUFFER_SIZE, msglen - l))  # this will not receive more bytes than required
            if len(recv_msg) == 0:  # connection closed or broken
                return None
            msg += recv_msg
        except OSError:  # connection closed or broken
            return None
        l = len(msg)
        # print(f"DEBUG2: l:{l}")
    # print("DEBUG: final data len:", l)
    # print(f"DEBUG: finished receiving after {time.time() - t_start}s.")
    send_ack(sock)
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
        print(f"INFO: connect() timed-out or failed, sleeping {cfg.WAIT_BEFORE_RECONNECTION}s")
        s.close()
        time.sleep(cfg.WAIT_BEFORE_RECONNECTION)
        return None
    s.settimeout(cfg.SOCKET_TIMEOUT_COMMUNICATE)
    return s


def accept_or_close_socket(s):
    """
    returns conn, addr
    None None in case of failure
    """
    conn = None
    try:
        conn, addr = s.accept()
        conn.settimeout(cfg.SOCKET_TIMEOUT_COMMUNICATE)
        return conn, addr
    except OSError:
        # print(f"INFO: accept() timed-out or failed, sleeping {WAIT_BEFORE_RECONNECTION}s")
        if conn is not None:
            conn.close()
        s.close()
        time.sleep(cfg.WAIT_BEFORE_RECONNECTION)
        return None, None


def select_and_send_or_close_socket(obj, conn):
    """
    Returns True if success
    False if disconnected (closes sockets)
    """
    _, wl, xl = select.select([], [conn], [conn], cfg.SELECT_TIMEOUT_OUTBOUND)  # select for writing
    if len(xl) != 0:
        print("INFO: error when writing, closing socket")
        conn.close()
        return False
    if len(wl) == 0:
        print("INFO: outbound select() timed out, closing socket")
        conn.close()
        return False
    elif not send_object(conn, obj):  # error or timeout
        print("INFO: send_object() failed, closing socket")
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
        # print(f"DEBUG: received obj:{obj}")
        return True, obj


# BUFFER: ===========================================

class Buffer:
    def __init__(self):
        self.memory = []
        self.stat_train_return = 0.0
        self.stat_test_return = 0.0

    def append_sample(self, sample):
        self.memory.append(sample)

    def clear(self):
        """
        Clears memory but keeps train and test returns
        """
        self.memory = []

    def __len__(self):
        return len(self.memory)

    def __iadd__(self, other):
        self.memory += other.memory
        self.stat_train_return = other.stat_train_return
        self.stat_test_return = other.stat_test_return
        return self


# REDIS SERVER: =====================================

class RedisServer:
    """
    This is the main server
    This lets 1 TrainerInterface and n RolloutWorkers connect
    This buffers experiences sent by RolloutWorkers
    This periodically sends the buffer to the TrainerInterface
    This also receives the weights from the TrainerInterface and broadcast them to the connected RolloutWorkers
    If localhost, the ip is localhost. Otherwise, it is the public ip and requires fort forwarding.
    """
    def __init__(self, samples_per_redis_batch=1000, localhost=True):
        self.__buffer = Buffer()
        self.__buffer_lock = Lock()
        self.__weights_lock = Lock()
        self.__weights = None
        self.samples_per_redis_batch = samples_per_redis_batch
        self.public_ip = get('http://api.ipify.org').text
        self.local_ip = socket.gethostbyname(socket.gethostname())
        self.ip = '127.0.0.1' if localhost else self.local_ip

        print(f"INFO REDIS: local IP: {self.local_ip}")
        print(f"INFO REDIS: public IP: {self.public_ip}")
        print(f"INFO REDIS: IP: {self.ip}")

        Thread(target=self.__rollout_workers_thread, args=(), kwargs={}, daemon=True).start()
        Thread(target=self.__trainer_thread, args=(), kwargs={}, daemon=True).start()

    def __trainer_thread(self, ):
        """
        This waits for a TrainerInterface to connect
        Then, this periodically sends the local buffer to the TrainerInterface (when data is available)
        When the TrainerInterface sends new weights, this broadcasts them to all connected RolloutWorkers
        """
        ack_time = time.time()
        wait_ack = False
        while True:  # main redis loop
            s = get_listening_socket(cfg.SOCKET_TIMEOUT_ACCEPT_TRAINER, self.ip, cfg.PORT_TRAINER)
            conn, addr = accept_or_close_socket(s)
            if conn is None:
                print("DEBUG: accept_or_close_socket failed in trainer thread")
                continue
            # last_ping = time.time()
            print(f"INFO TRAINER THREAD: redis connected by trainer at address {addr}")
            # Here we could spawn a Trainer communication thread, but since there is only one trainer we move on
            i = 0
            while True:
                # ping client
                # if time.time() - last_ping >= PING_INTERVAL:
                #     print("INFO: sending ping to trainer")
                #     if ping_pong(conn):
                #         last_ping = time.time()
                #     else:
                #         print("INFO: ping to trainer client failed")
                #         break
                # send samples
                self.__buffer_lock.acquire()  # BUFFER LOCK.............................................................
                if len(self.__buffer) >= self.samples_per_redis_batch:
                    if not wait_ack:
                        obj = self.__buffer
                        print(f"INFO: sending obj {i}")
                        if select_and_send_or_close_socket(obj, conn):
                            wait_ack = True
                            ack_time = time.time()
                        else:
                            print("INFO: failed sending object to trainer")
                            self.__buffer_lock.release()
                            break
                        self.__buffer.clear()
                    else:
                        elapsed = time.time() - ack_time
                        print(f"WARNING: object ready but ACK from last transmission not received. Elapsed:{elapsed}s")
                        if elapsed >= cfg.ACK_TIMEOUT_REDIS_TO_TRAINER:
                            print("INFO: ACK timed-out, breaking connection")
                            self.__buffer_lock.release()
                            break
                self.__buffer_lock.release()  # END BUFFER LOCK.........................................................
                # checks for weights
                success, obj = poll_and_recv_or_close_socket(conn)
                if not success:
                    print("DEBUG: poll failed in trainer thread")
                    break
                elif obj is not None and obj != 'ACK':
                    print(f"DEBUG INFO: trainer thread received obj")
                    self.__weights_lock.acquire()  # WEIGHTS LOCK.......................................................
                    self.__weights = obj
                    self.__weights_lock.release()  # END WEIGHTS LOCK...................................................
                elif obj == 'ACK':
                    wait_ack = False
                    print(f"INFO: transfer acknowledgment received after {time.time() - ack_time}s")
                time.sleep(cfg.LOOP_SLEEP_TIME)  # TODO: adapt
                i += 1
            s.close()

    def __rollout_workers_thread(self):
        """
        This waits for new potential RolloutWorkers to connect
        When a new RolloutWorker connects, this instantiates a new thread to handle it
        """
        while True:  # main redis loop
            s = get_listening_socket(cfg.SOCKET_TIMEOUT_ACCEPT_ROLLOUT, self.ip, cfg.PORT_ROLLOUT)
            conn, addr = accept_or_close_socket(s)
            if conn is None:
                # print("DEBUG: accept_or_close_socket failed in workers thread")
                continue
            print(f"INFO WORKERS THREAD: redis connected by worker at address {addr}")
            Thread(target=self.__rollout_worker_thread, args=(conn, ), kwargs={}, daemon=True).start()  # we don't keep track of this for now
            s.close()

    def __rollout_worker_thread(self, conn):
        """
        Thread handling connection to a single RolloutWorker
        """
        # last_ping = time.time()
        ack_time = time.time()
        wait_ack = False
        while True:
            # ping client
            # if time.time() - last_ping >= PING_INTERVAL:
            #     print("INFO: sending ping to worker")
            #     if ping_pong(conn):
            #         last_ping = time.time()
            #     else:
            #         print("INFO: ping to trainer client failed")
            #         break
            # send weights
            self.__weights_lock.acquire()  # WEIGHTS LOCK...............................................................
            if self.__weights is not None:  # new weigths
                if not wait_ack:
                    obj = self.__weights
                    if select_and_send_or_close_socket(obj, conn):
                        ack_time = time.time()
                        wait_ack = True
                    else:
                        self.__weights_lock.release()
                        print("DEBUG: select_and_send_or_close_socket failed in worker thread")
                        break
                    self.__weights = None
                else:
                    elapsed = time.time() - ack_time
                    print(f"WARNING: object ready but ACK from last transmission not received. Elapsed:{elapsed}s")
                    if elapsed >= cfg.ACK_TIMEOUT_REDIS_TO_WORKER:
                        print("INFO: ACK timed-out, breaking connection")
                        self.__weights_lock.release()
                        break
            self.__weights_lock.release()  # END WEIGHTS LOCK...........................................................
            # checks for samples
            success, obj = poll_and_recv_or_close_socket(conn)
            if not success:
                print("DEBUG: poll failed in rollout thread")
                break
            elif obj is not None and obj != 'ACK':
                print(f"DEBUG INFO: rollout worker thread received obj")
                self.__buffer_lock.acquire()  # BUFFER LOCK.............................................................
                self.__buffer += obj  # concat worker batch to local batch
                self.__buffer_lock.release()  # END BUFFER LOCK.........................................................
            elif obj == 'ACK':
                wait_ack = False
                print(f"INFO: transfer acknowledgment received after {time.time() - ack_time}s")
            time.sleep(cfg.LOOP_SLEEP_TIME)  # TODO: adapt


# TRAINER: ==========================================

class TrainerInterface:
    """
    This is the trainer's network interface
    This connects to the redis server
    This receives samples batches and sends new weights
    """
    def __init__(self,
                 redis_ip=None,
                 model_path=cfg.MODEL_PATH_TRAINER):
        self.__buffer_lock = Lock()
        self.__weights_lock = Lock()
        self.__weights = None
        self.__buffer = Buffer()
        self.model_path = model_path
        self.public_ip = get('http://api.ipify.org').text
        self.local_ip = socket.gethostbyname(socket.gethostname())
        self.redis_ip = redis_ip if redis_ip is not None else '127.0.0.1'

        print(f"local IP: {self.local_ip}")
        print(f"public IP: {self.public_ip}")
        print(f"redis IP: {self.redis_ip}")

        Thread(target=self.__run_thread, args=(), kwargs={}, daemon=True).start()

    def __run_thread(self):
        """
        Trainer interface thread
        """
        ack_time = time.time()
        wait_ack = False
        while True:  # main client loop
            s = get_connected_socket(cfg.SOCKET_TIMEOUT_CONNECT_TRAINER, self.redis_ip, cfg.PORT_TRAINER)
            if s is None:
                print("DEBUG: get_connected_socket failed in TrainerInterface thread")
                continue
            while True:
                # send weights
                self.__weights_lock.acquire()  # WEIGHTS LOCK...........................................................
                if self.__weights is not None:  # new weights
                    if not wait_ack:
                        obj = self.__weights
                        if select_and_send_or_close_socket(obj, s):
                            ack_time = time.time()
                            wait_ack = True
                        else:
                            self.__weights_lock.release()
                            print("DEBUG: select_and_send_or_close_socket failed in TrainerInterface")
                            break
                        self.__weights = None
                    else:
                        elapsed = time.time() - ack_time
                        print(f"WARNING: object ready but ACK from last transmission not received. Elapsed:{elapsed}s")
                        if elapsed >= cfg.ACK_TIMEOUT_TRAINER_TO_REDIS:
                            print("INFO: ACK timed-out, breaking connection")
                            self.__weights_lock.release()
                            break
                self.__weights_lock.release()  # END WEIGHTS LOCK.......................................................
                # checks for samples batch
                success, obj = poll_and_recv_or_close_socket(s)
                if not success:
                    print("DEBUG: poll failed in TrainerInterface thread")
                    break
                elif obj is not None and obj != 'ACK':  # received buffer
                    print(f"DEBUG INFO: trainer interface received obj")
                    self.__buffer_lock.acquire()  # BUFFER LOCK.........................................................
                    self.__buffer += obj
                    self.__buffer_lock.release()  # END BUFFER LOCK.....................................................
                elif obj == 'ACK':
                    wait_ack = False
                    print(f"INFO: transfer acknowledgment received after {time.time() - ack_time}s")
                time.sleep(cfg.LOOP_SLEEP_TIME)  # TODO: adapt
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
        returns a copy of the TrainerInterface's local buffer, and clears it
        """
        self.__buffer_lock.acquire()  # BUFFER LOCK.....................................................................
        buffer_copy = deepcopy(self.__buffer)
        self.__buffer.clear()
        self.__buffer_lock.release()  # END BUFFER LOCK.................................................................
        return buffer_copy


# ROLLOUT WORKER: ===================================

class RolloutWorker:
    def __init__(self,
                 env_id,
                 actor_module_cls,
                 # obs_space,
                 # act_space,
                 get_local_buffer_sample: callable,
                 device="cpu",
                 redis_ip=None,
                 samples_per_worker_batch=1000,
                 model_path=cfg.MODEL_PATH_WORKER,
                 obs_preprocessor: callable = None
                 ):
        self.obs_preprocessor = obs_preprocessor
        self.get_local_buffer_sample = get_local_buffer_sample
        self.env = UntouchedGymEnv(id=env_id, gym_kwargs={"config": cfg.CONFIG_DICT})
        obs_space = self.env.observation_space
        act_space = self.env.action_space
        self.model_path = model_path
        self.actor = actor_module_cls(obs_space, act_space).to(device)
        if os.path.isfile(self.model_path):
            self.actor.load_state_dict(torch.load(self.model_path))
        self.device = device
        self.buffer = Buffer()
        self.__buffer = Buffer()  # deepcopy for sending
        self.__buffer_lock = Lock()
        self.__weights = None
        self.__weights_lock = Lock()
        self.samples_per_worker_batch = samples_per_worker_batch

        self.public_ip = get('http://api.ipify.org').text
        self.local_ip = socket.gethostbyname(socket.gethostname())
        self.redis_ip = redis_ip if redis_ip is not None else '127.0.0.1'

        print(f"local IP: {self.local_ip}")
        print(f"public IP: {self.public_ip}")
        print(f"redis IP: {self.redis_ip}")

        Thread(target=self.__run_thread, args=(), kwargs={}, daemon=True).start()

    def __run_thread(self):
        """
        Redis thread
        """
        ack_time = time.time()
        wait_ack = False
        while True:  # main client loop
            s = get_connected_socket(cfg.SOCKET_TIMEOUT_CONNECT_ROLLOUT, self.redis_ip, cfg.PORT_ROLLOUT)
            if s is None:
                print("DEBUG: get_connected_socket failed in worker")
                continue
            while True:
                # send buffer
                self.__buffer_lock.acquire()  # BUFFER LOCK.............................................................
                if len(self.__buffer) >= self.samples_per_worker_batch:  # a new batch is available
                    if not wait_ack:
                        obj = self.__buffer
                        if select_and_send_or_close_socket(obj, s):
                            ack_time = time.time()
                            wait_ack = True
                        else:
                            self.__buffer_lock.release()
                            print("DEBUG: select_and_send_or_close_socket failed in worker")
                            break
                        self.__buffer.clear()  # empty sent batch
                    else:
                        elapsed = time.time() - ack_time
                        print(f"WARNING: object ready but ACK from last transmission not received. Elapsed:{elapsed}s")
                        if elapsed >= cfg.ACK_TIMEOUT_WORKER_TO_REDIS:
                            print("INFO: ACK timed-out, breaking connection")
                            self.__buffer_lock.release()
                            break
                self.__buffer_lock.release()  # END BUFFER LOCK.........................................................
                # checks for new weights
                success, obj = poll_and_recv_or_close_socket(s)
                if not success:
                    print(f"INFO: rollout worker poll failed")
                    break
                elif obj is not None and obj != 'ACK':
                    print(f"DEBUG INFO: rollout worker received obj")
                    self.__weights_lock.acquire()  # WEIGHTS LOCK.......................................................
                    self.__weights = obj
                    self.__weights_lock.release()  # END WEIGHTS LOCK...................................................
                elif obj == 'ACK':
                    wait_ack = False
                    print(f"INFO: transfer acknowledgment received after {time.time() - ack_time}s")
                time.sleep(cfg.LOOP_SLEEP_TIME)  # TODO: adapt
            s.close()

    def act(self, obs, train=False):
        """
        converts inputs to torch tensors and converts outputs to numpy arrays
        """
        if self.obs_preprocessor is not None:
            obs = self.obs_preprocessor(obs)
        obs = collate([obs], device=self.device)
        with torch.no_grad():
            action_distribution = self.actor(obs)
            action = action_distribution.sample() if train else action_distribution.sample_deterministic()
        action, = partition(action)
        return action

    def reset(self, train, collect_samples):
        act = self.env.default_action.astype(np.float32)
        obs = self.env.reset()
        if collect_samples:
            sample = self.get_local_buffer_sample(act, obs, 0.0, False, {})
            self.buffer.append_sample(sample)
        return obs

    def step(self, obs, train, collect_samples):
        act = self.act(obs, train=train)
        obs, rew, done, info = self.env.step(act)
        if collect_samples:
            sample = self.get_local_buffer_sample(act, obs, rew, done, info)
            self.buffer.append_sample(sample)  # CAUTION: in the buffer, act is for the PREVIOUS transition (act, obs(act))
        return obs, rew, done, info

    def collect_train_episode(self, max_samples):
        """
        collects a maximum of n training transitions (from reset to done)
        stores episode and train return in the local buffer of the worker
        """
        ret = 0.0
        obs = self.reset(train=True, collect_samples=True)
        for _ in range(max_samples):
            obs, rew, done, info = self.step(obs=obs, train=True, collect_samples=True)
            ret += rew
            if done:
                break
        self.buffer.stat_train_return = ret
        print(f"DEBUG:self.buffer.stat_train_return:{self.buffer.stat_train_return}")

    def run_test_episode(self, max_samples):
        """
        collects a maximum of n test transitions (from reset to done)
        stores test return in the local buffer of the worker
        """
        ret = 0.0
        obs = self.reset(train=False, collect_samples=False)
        for _ in range(max_samples):
            obs, rew, done, info = self.step(obs=obs, train=False, collect_samples=False)
            ret += rew
            if done:
                break
        self.buffer.stat_test_return = ret
        print(f"DEBUG:self.buffer.stat_test_return:{self.buffer.stat_test_return}")

    def run(self, test_episode_interval=20):  # TODO: check number of collected samples are collected before sending
        episode = 0
        while True:
            if episode % test_episode_interval == 0:
                print("INFO: running test episode")
                self.run_test_episode(self.samples_per_worker_batch)
            print("INFO: collecting train episode")
            self.collect_train_episode(self.samples_per_worker_batch)
            print("INFO: copying buffer for sending")
            self.send_and_clear_buffer()
            print("INFO: checking for new weights")
            self.update_actor_weights()
            episode += 1

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
            print("INFO: model weights have been updated")
            self.__weights = None
        self.__weights_lock.release()  # END WEIGHTS LOCK...............................................................

