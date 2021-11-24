# standard library imports
import datetime
import os
import pickle
import select
import socket
import time
from copy import deepcopy
from threading import Lock, Thread

# third-party imports
import numpy as np
import torch
from requests import get

# local imports
import tmrl.config.config_constants as cfg
from tmrl.sac_models import ActorModule
from tmrl.util import collate
import logging
# PRINT: ============================================


def print_with_timestamp(s):
    x = datetime.datetime.now()
    sx = x.strftime("%x %X ")
    logging.info(sx + str(s))


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
        print_with_timestamp("socket error/timeout while sending PING")
        sock.close()
        return False
    send_ping(sock)
    rl, _, xl = select.select([sock], [], [sock], cfg.SELECT_TIMEOUT_PING_PONG)  # select for reading
    if len(xl) != 0 or len(rl) == 0:
        print_with_timestamp("socket error/timeout while waiting for PONG")
        sock.close()
        return False
    obj = recv_object(sock)
    if obj == 'PINGPONG':
        return True
    else:
        print_with_timestamp("PINGPONG received an object that is not PING or PONG")
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
        if cfg.PRINT_BYTESIZES:
            print_with_timestamp(f"Sending {len(msg)} bytes.")
    try:
        sock.sendall(msg)
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
    if msg[:4] == b'PING' or msg[:4] == b'PONG':
        if msg[:4] == b'PING':
            send_pong(sock)
        return 'PINGPONG'
    if msg[:3] == b'ACK':
        return 'ACK'
    msglen = int(msg[:cfg.HEADER_SIZE])
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
    send_ack(sock)
    return pickle.loads(msg)


def get_listening_socket(timeout, ip_bind, port_bind):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # to reuse address on Linux
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
        print_with_timestamp(f"connect() timed-out or failed, sleeping {cfg.WAIT_BEFORE_RECONNECTION}s")
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
    print_with_timestamp(f"start select")
    _, wl, xl = select.select([], [conn], [conn], cfg.SELECT_TIMEOUT_OUTBOUND)  # select for writing
    print_with_timestamp(f"end select")
    if len(xl) != 0:
        print_with_timestamp("error when writing, closing socket")
        conn.close()
        return False
    if len(wl) == 0:
        print_with_timestamp("outbound select() timed out, closing socket")
        conn.close()
        return False
    elif not send_object(conn, obj):  # error or timeout
        print_with_timestamp("send_object() failed, closing socket")
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
        print_with_timestamp("error when polling, closing sockets")
        conn.close()
        return False, None
    if len(rl) == 0:  # nothing in the recv buffer
        return True, None
    obj = recv_object(conn)
    if obj is None:  # socket error
        print_with_timestamp("error when receiving object, closing sockets")
        conn.close()
        return False, None
    elif obj == 'PINGPONG':
        return True, None
    else:
        return True, obj


# BUFFER: ===========================================


class Buffer:
    def __init__(self, maxlen=cfg.BUFFERS_MAXLEN):
        self.memory = []
        self.stat_train_return = 0.0
        self.stat_test_return = 0.0
        self.stat_train_steps = 0
        self.stat_test_steps = 0
        self.maxlen = maxlen

    def clip_to_maxlen(self):
        lenmem = len(self.memory)
        if lenmem > self.maxlen:
            print_with_timestamp("buffer overflow. Discarding old samples.")
            self.memory = self.memory[(lenmem - self.maxlen):]

    def append_sample(self, sample):
        self.memory.append(sample)
        self.clip_to_maxlen()

    def clear(self):
        """
        Clears memory but keeps train and test returns
        """
        self.memory = []

    def __len__(self):
        return len(self.memory)

    def __iadd__(self, other):
        self.memory += other.memory
        self.clip_to_maxlen()
        self.stat_train_return = other.stat_train_return
        self.stat_test_return = other.stat_test_return
        self.stat_train_steps = other.stat_train_steps
        self.stat_test_steps = other.stat_test_steps
        return self


# SERVER SERVER: =====================================


class Server:
    """
    This is the main server
    This lets 1 TrainerInterface and n RolloutWorkers connect
    This buffers experiences sent by RolloutWorkers
    This periodically sends the buffer to the TrainerInterface
    This also receives the weights from the TrainerInterface and broadcast them to the connected RolloutWorkers
    If trainer_on_localhost is True, the server only listens on trainer_on_localhost. Then the trainer is expected to talk on trainer_on_localhost.
    Otherwise, the server also listens to the local ip and the trainer is expected to talk on the local ip (port forwarding).
    """
    def __init__(self, samples_per_server_packet=1000):
        self.__buffer = Buffer()
        self.__buffer_lock = Lock()
        self.__weights_lock = Lock()
        self.__weights = None
        self.__weights_id = 0  # this increments each time new weights are received
        self.samples_per_server_batch = samples_per_server_packet
        self.public_ip = get('http://api.ipify.org').text
        self.local_ip = socket.gethostbyname(socket.gethostname())

        print_with_timestamp(f"INFO SERVER: local IP: {self.local_ip}")
        print_with_timestamp(f"INFO SERVER: public IP: {self.public_ip}")

        Thread(target=self.__rollout_workers_thread, args=('', ), kwargs={}, daemon=True).start()
        Thread(target=self.__trainers_thread, args=('', ), kwargs={}, daemon=True).start()

    def __trainers_thread(self, ip):
        """
        This waits for new potential Trainers to connect
        When a new Trainer connects, this instantiates a new thread to handle it
        """
        while True:  # main server loop
            s = get_listening_socket(cfg.SOCKET_TIMEOUT_ACCEPT_TRAINER, ip, cfg.PORT_TRAINER)
            conn, addr = accept_or_close_socket(s)
            if conn is None:
                continue
            print_with_timestamp(f"INFO TRAINERS THREAD: server connected by trainer at address {addr}")
            Thread(target=self.__trainer_thread, args=(conn, ), kwargs={}, daemon=True).start()  # we don't keep track of this for now
            s.close()

    def __trainer_thread(self, conn):
        """
        This periodically sends the local buffer to the TrainerInterface (when data is available)
        When the TrainerInterface sends new weights, this broadcasts them to all connected RolloutWorkers
        """
        ack_time = time.time()
        wait_ack = False
        while True:
            # send samples
            self.__buffer_lock.acquire()  # BUFFER LOCK.............................................................
            if len(self.__buffer) >= self.samples_per_server_batch:
                if not wait_ack:
                    obj = self.__buffer
                    if select_and_send_or_close_socket(obj, conn):
                        wait_ack = True
                        ack_time = time.time()
                    else:
                        print_with_timestamp("failed sending object to trainer")
                        self.__buffer_lock.release()
                        break
                    self.__buffer.clear()
                else:
                    elapsed = time.time() - ack_time
                    print_with_timestamp(f"CAUTION: object ready but ACK from last transmission not received. Elapsed:{elapsed}s")
                    if elapsed >= cfg.ACK_TIMEOUT_SERVER_TO_TRAINER:
                        print_with_timestamp("ACK timed-out, breaking connection")
                        self.__buffer_lock.release()
                        wait_ack = False
                        break
            self.__buffer_lock.release()  # END BUFFER LOCK.........................................................
            # checks for weights
            success, obj = poll_and_recv_or_close_socket(conn)
            if not success:
                print_with_timestamp("poll failed in trainer thread")
                break
            elif obj is not None and obj != 'ACK':
                print_with_timestamp(f"trainer thread received obj")
                self.__weights_lock.acquire()  # WEIGHTS LOCK.......................................................
                self.__weights = obj
                self.__weights_id += 1
                self.__weights_lock.release()  # END WEIGHTS LOCK...................................................
            elif obj == 'ACK':
                wait_ack = False
                print_with_timestamp(f"transfer acknowledgment received after {time.time() - ack_time}s")
            time.sleep(cfg.LOOP_SLEEP_TIME)  # TODO: adapt

    def __rollout_workers_thread(self, ip):
        """
        This waits for new potential RolloutWorkers to connect
        When a new RolloutWorker connects, this instantiates a new thread to handle it
        """
        while True:  # main server loop
            s = get_listening_socket(cfg.SOCKET_TIMEOUT_ACCEPT_ROLLOUT, ip, cfg.PORT_ROLLOUT)
            conn, addr = accept_or_close_socket(s)
            if conn is None:
                continue
            print_with_timestamp(f"INFO WORKERS THREAD: server connected by worker at address {addr}")
            Thread(target=self.__rollout_worker_thread, args=(conn, ), kwargs={}, daemon=True).start()  # we don't keep track of this for now
            s.close()

    def __rollout_worker_thread(self, conn):
        """
        Thread handling connection to a single RolloutWorker
        """
        # last_ping = time.time()
        worker_weights_id = 0
        ack_time = time.time()
        wait_ack = False
        while True:
            # send weights
            self.__weights_lock.acquire()  # WEIGHTS LOCK...............................................................
            if worker_weights_id != self.__weights_id:  # new weigths
                if not wait_ack:
                    obj = self.__weights
                    if select_and_send_or_close_socket(obj, conn):
                        ack_time = time.time()
                        wait_ack = True
                    else:
                        self.__weights_lock.release()
                        print_with_timestamp("select_and_send_or_close_socket failed in worker thread")
                        break
                    worker_weights_id = self.__weights_id
                else:
                    elapsed = time.time() - ack_time
                    print_with_timestamp(f"object ready but ACK from last transmission not received. Elapsed:{elapsed}s")
                    if elapsed >= cfg.ACK_TIMEOUT_SERVER_TO_WORKER:
                        print_with_timestamp("ACK timed-out, breaking connection")
                        self.__weights_lock.release()
                        # wait_ack = False  # not needed since we end the thread
                        break
            self.__weights_lock.release()  # END WEIGHTS LOCK...........................................................
            # checks for samples
            success, obj = poll_and_recv_or_close_socket(conn)
            if not success:
                print_with_timestamp("poll failed in rollout thread")
                break
            elif obj is not None and obj != 'ACK':
                print_with_timestamp(f"rollout worker thread received obj")
                self.__buffer_lock.acquire()  # BUFFER LOCK.............................................................
                self.__buffer += obj  # concat worker batch to local batch
                self.__buffer_lock.release()  # END BUFFER LOCK.........................................................
            elif obj == 'ACK':
                wait_ack = False
                print_with_timestamp(f"transfer acknowledgment received after {time.time() - ack_time}s")
            time.sleep(cfg.LOOP_SLEEP_TIME)  # TODO: adapt


# TRAINER: ==========================================


class TrainerInterface:
    """
    This is the trainer's network interface
    This connects to the server
    This receives samples batches and sends new weights
    """
    def __init__(self, server_ip=None, model_path=cfg.MODEL_PATH_TRAINER):
        self.__buffer_lock = Lock()
        self.__weights_lock = Lock()
        self.__weights = None
        self.__buffer = Buffer()
        self.model_path = model_path
        self.public_ip = get('http://api.ipify.org').text
        self.local_ip = socket.gethostbyname(socket.gethostname())
        self.server_ip = server_ip if server_ip is not None else '127.0.0.1'
        self.recv_tiemout = cfg.RECV_TIMEOUT_TRAINER_FROM_SERVER

        print_with_timestamp(f"local IP: {self.local_ip}")
        print_with_timestamp(f"public IP: {self.public_ip}")
        print_with_timestamp(f"server IP: {self.server_ip}")

        Thread(target=self.__run_thread, args=(), kwargs={}, daemon=True).start()

    def __run_thread(self):
        """
        Trainer interface thread
        """
        while True:  # main client loop
            ack_time = time.time()
            recv_time = time.time()
            wait_ack = False
            s = get_connected_socket(cfg.SOCKET_TIMEOUT_CONNECT_TRAINER, self.server_ip, cfg.PORT_TRAINER)
            if s is None:
                print_with_timestamp("get_connected_socket failed in TrainerInterface thread")
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
                            print_with_timestamp("select_and_send_or_close_socket failed in TrainerInterface")
                            break
                        self.__weights = None
                    else:
                        elapsed = time.time() - ack_time
                        print_with_timestamp(f"CAUTION: object ready but ACK from last transmission not received. Elapsed:{elapsed}s")
                        if elapsed >= cfg.ACK_TIMEOUT_TRAINER_TO_SERVER:
                            print_with_timestamp("ACK timed-out, breaking connection")
                            self.__weights_lock.release()
                            wait_ack = False
                            break
                self.__weights_lock.release()  # END WEIGHTS LOCK.......................................................
                # checks for samples batch
                success, obj = poll_and_recv_or_close_socket(s)
                if not success:
                    print_with_timestamp("poll failed in TrainerInterface thread")
                    break
                elif obj is not None and obj != 'ACK':  # received buffer
                    print_with_timestamp(f"trainer interface received obj")
                    recv_time = time.time()
                    self.__buffer_lock.acquire()  # BUFFER LOCK.........................................................
                    self.__buffer += obj
                    self.__buffer_lock.release()  # END BUFFER LOCK.....................................................
                elif obj == 'ACK':
                    wait_ack = False
                    print_with_timestamp(f"transfer acknowledgment received after {time.time() - ack_time}s")
                elif time.time() - recv_time > self.recv_tiemout:
                    print_with_timestamp(f"Timeout in TrainerInterface, not received anything for too long")
                    break
                time.sleep(cfg.LOOP_SLEEP_TIME)  # TODO: adapt
            s.close()

    def broadcast_model(self, model: ActorModule):
        """
        model must be an ActorModule (sac_models.py)
        broadcasts the model's weights to all connected RolloutWorkers
        """
        t0 = time.time()
        self.__weights_lock.acquire()  # WEIGHTS LOCK...................................................................
        t1 = time.time()
        torch.save(model.state_dict(), self.model_path)
        t2 = time.time()
        with open(self.model_path, 'rb') as f:
            self.__weights = f.read()
        t3 = time.time()
        self.__weights_lock.release()  # END WEIGHTS LOCK...............................................................
        print_with_timestamp(f"broadcast_model: lock acquire: {t1 - t0}s, save dict: {t2 - t1}s, read dict: {t3 - t2}s")

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
    def __init__(
            self,
            env_cls,
            actor_module_cls,
            get_local_buffer_sample: callable,
            device="cpu",
            server_ip=None,
            samples_per_worker_packet=1000,  # The RolloutWorker waits for this number of samples before sending
            max_samples_per_episode=1000000,  # If the episode is longer than this, it is reset by the RolloutWorker
            model_path=cfg.MODEL_PATH_WORKER,
            obs_preprocessor: callable = None,
            crc_debug=False,
            model_path_history=cfg.MODEL_PATH_SAVE_HISTORY,
            model_history=cfg.MODEL_HISTORY,  # if 0, doesn't save model history, else, the model is saved every model_history episode
            standalone=False,  # if True, the worker will not try to connect to a server but only
    ):
        self.obs_preprocessor = obs_preprocessor
        self.get_local_buffer_sample = get_local_buffer_sample
        self.env = env_cls()
        obs_space = self.env.observation_space
        act_space = self.env.action_space
        self.model_path = model_path
        self.model_path_history = model_path_history
        self.actor = actor_module_cls(obs_space, act_space).to(device)
        self.device = device
        self.standalone = standalone
        if os.path.isfile(self.model_path):
            self.actor.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.buffer = Buffer()
        self.__buffer = Buffer()  # deepcopy for sending
        self.__buffer_lock = Lock()
        self.__weights = None
        self.__weights_lock = Lock()
        self.samples_per_worker_batch = samples_per_worker_packet
        self.max_samples_per_episode = max_samples_per_episode
        self.crc_debug = crc_debug
        self.model_history = model_history
        self._cur_hist_cpt = 0

        self.public_ip = get('http://api.ipify.org').text
        self.local_ip = socket.gethostbyname(socket.gethostname())
        self.server_ip = server_ip if server_ip is not None else '127.0.0.1'
        self.recv_timeout = cfg.RECV_TIMEOUT_WORKER_FROM_SERVER

        print_with_timestamp(f"local IP: {self.local_ip}")
        print_with_timestamp(f"public IP: {self.public_ip}")
        print_with_timestamp(f"server IP: {self.server_ip}")

        if not self.standalone:
            Thread(target=self.__run_thread, args=(), kwargs={}, daemon=True).start()

    def __run_thread(self):
        """
        Redis thread
        """
        while True:  # main client loop
            ack_time = time.time()
            recv_time = time.time()
            wait_ack = False
            s = get_connected_socket(cfg.SOCKET_TIMEOUT_CONNECT_ROLLOUT, self.server_ip, cfg.PORT_ROLLOUT)
            if s is None:
                print_with_timestamp("get_connected_socket failed in worker")
                continue
            while True:
                # send buffer
                self.__buffer_lock.acquire()  # BUFFER LOCK.............................................................
                if len(self.__buffer) >= self.samples_per_worker_batch:  # a new batch is available
                    print_with_timestamp("new batch available")
                    if not wait_ack:
                        obj = self.__buffer
                        if select_and_send_or_close_socket(obj, s):
                            ack_time = time.time()
                            wait_ack = True
                        else:
                            self.__buffer_lock.release()
                            print_with_timestamp("select_and_send_or_close_socket failed in worker")
                            break
                        self.__buffer.clear()  # empty sent batch
                    else:
                        elapsed = time.time() - ack_time
                        print_with_timestamp(f"CAUTION: object ready but ACK from last transmission not received. Elapsed:{elapsed}s")
                        if elapsed >= cfg.ACK_TIMEOUT_WORKER_TO_SERVER:
                            print_with_timestamp("ACK timed-out, breaking connection")
                            self.__buffer_lock.release()
                            wait_ack = False
                            break
                self.__buffer_lock.release()  # END BUFFER LOCK.........................................................
                # checks for new weights
                success, obj = poll_and_recv_or_close_socket(s)
                if not success:
                    print_with_timestamp(f"rollout worker poll failed")
                    break
                elif obj is not None and obj != 'ACK':
                    print_with_timestamp(f"rollout worker received obj")
                    recv_time = time.time()
                    self.__weights_lock.acquire()  # WEIGHTS LOCK.......................................................
                    self.__weights = obj
                    self.__weights_lock.release()  # END WEIGHTS LOCK...................................................
                elif obj == 'ACK':
                    wait_ack = False
                    print_with_timestamp(f"transfer acknowledgment received after {time.time() - ack_time}s")
                elif time.time() - recv_time > self.recv_timeout:
                    print_with_timestamp(f"Timeout in RolloutWorker, not received anything for too long")
                    break
                time.sleep(cfg.LOOP_SLEEP_TIME)  # TODO: adapt
            s.close()

    def act(self, obs, deterministic=False):
        """
        converts inputs to torch tensors and converts outputs to numpy arrays
        """
        if self.obs_preprocessor is not None:
            obs = self.obs_preprocessor(obs)
        obs = collate([obs], device=self.device)
        with torch.no_grad():
            action = self.actor.act(obs, deterministic=deterministic)
            # action = action_distribution.sample() if train else action_distribution.sample_deterministic()
        # action, = partition(action)
        return action

    def reset(self, collect_samples):
        obs = None
        act = self.env.default_action.astype(np.float32)
        new_obs = self.env.reset()
        # if self.obs_preprocessor is not None:
        #     new_obs = self.obs_preprocessor(new_obs)
        rew = 0.0
        done = False
        info = {}
        if collect_samples:
            if self.crc_debug:
                info['crc_sample'] = (obs, act, new_obs, rew, done)
            sample = self.get_local_buffer_sample(act, new_obs, rew, done, info)
            self.buffer.append_sample(sample)
        return new_obs

    def step(self, obs, deterministic, collect_samples, last_step=False):
        act = self.act(obs, deterministic=deterministic)
        new_obs, rew, done, info = self.env.step(act)
        # if self.obs_preprocessor is not None:
        #     new_obs = self.obs_preprocessor(new_obs)
        if collect_samples:
            stored_done = done
            if last_step and not done:  # ignore done when stopped by step limit
                info["__no_done"] = True
            if "__no_done" in info:
                stored_done = False
            if self.crc_debug:
                info['crc_sample'] = (obs, act, new_obs, rew, stored_done)
            sample = self.get_local_buffer_sample(act, new_obs, rew, stored_done, info)
            self.buffer.append_sample(sample)  # CAUTION: in the buffer, act is for the PREVIOUS transition (act, obs(act))
        return new_obs, rew, done, info

    def collect_train_episode(self, max_samples):
        """
        collects a maximum of n training transitions (from reset to done)
        stores episode and train return in the local buffer of the worker
        """
        ret = 0.0
        steps = 0
        obs = self.reset(collect_samples=True)
        for i in range(max_samples):
            obs, rew, done, info = self.step(obs=obs, deterministic=False, collect_samples=True, last_step=i == max_samples - 1)
            ret += rew
            steps += 1
            if done:
                break
        self.buffer.stat_train_return = ret
        self.buffer.stat_train_steps = steps

    def run_episodes(self, max_samples, train=False):
        """
        collects a maximum of n test transitions (from reset to done)
        stores test return in the local buffer of the worker
        """
        while True:
            ret = 0.0
            steps = 0
            obs = self.reset(collect_samples=False)
            for _ in range(max_samples):
                obs, rew, done, info = self.step(obs=obs, deterministic=not train, collect_samples=False)
                ret += rew
                steps += 1
                if done:
                    break
            self.buffer.stat_test_return = ret
            self.buffer.stat_test_steps = steps

    def run_test_episode(self, max_samples):
        """
        collects a maximum of n test transitions (from reset to done)
        stores test return in the local buffer of the worker
        """
        ret = 0.0
        steps = 0
        obs = self.reset(collect_samples=False)
        for _ in range(max_samples):
            obs, rew, done, info = self.step(obs=obs, deterministic=True, collect_samples=False)
            ret += rew
            steps += 1
            if done:
                break
        self.buffer.stat_test_return = ret
        self.buffer.stat_test_steps = steps

    def run(self, test_episode_interval=20):  # TODO: check number of collected samples are collected before sending
        episode = 0
        while True:
            if episode % test_episode_interval == 0 and not self.crc_debug:
                print_with_timestamp("running test episode")
                self.run_test_episode(self.max_samples_per_episode)
            print_with_timestamp("collecting train episode")
            self.collect_train_episode(self.max_samples_per_episode)
            print_with_timestamp("copying buffer for sending")
            self.send_and_clear_buffer()
            print_with_timestamp("checking for new weights")
            self.update_actor_weights()
            episode += 1
            # if self.crc_debug:
            #     break

    def profile_step(self, nb_steps=100):
        import torch.autograd.profiler as profiler
        obs = self.reset(collect_samples=True)
        use_cuda = True if self.device == 'cuda' else False
        print_with_timestamp(f"use_cuda:{use_cuda}")
        with profiler.profile(record_shapes=True, use_cuda=use_cuda) as prof:
            obs = collate([obs], device=self.device)
            with profiler.record_function("pytorch_profiler"):
                with torch.no_grad():
                    action_distribution = self.actor(obs)
                    action = action_distribution.sample()
        print_with_timestamp(prof.key_averages().table(row_limit=20, sort_by="cpu_time_total"))

    def run_env_benchmark(self, nb_steps, deterministic=False):
        """
        This method is only compatible with rtgym environments
        """
        obs = self.reset(collect_samples=False)
        for _ in range(nb_steps):
            obs, rew, done, info = self.step(obs=obs, deterministic=deterministic, collect_samples=False)
            if done:
                obs = self.reset(collect_samples=False)
        print_with_timestamp(f"Benchmark results:\n{self.env.benchmarks()}")

    def send_and_clear_buffer(self):
        self.__buffer_lock.acquire()  # BUFFER LOCK.....................................................................
        self.__buffer += self.buffer
        self.__buffer_lock.release()  # END BUFFER LOCK.................................................................
        self.buffer.clear()

    def update_actor_weights(self):
        """
        updates the model with new weights from the trainer when available
        """
        self.__weights_lock.acquire()  # WEIGHTS LOCK...................................................................
        if self.__weights is not None:  # new weights available
            with open(self.model_path, 'wb') as f:
                f.write(self.__weights)
            if self.model_history:
                self._cur_hist_cpt += 1
                if self._cur_hist_cpt == self.model_history:
                    x = datetime.datetime.now()
                    with open(self.model_path_history + str(x.strftime("%d_%m_%Y_%H_%M_%S")) + ".pth", 'wb') as f:
                        f.write(self.__weights)
                    self._cur_hist_cpt = 0
                    print_with_timestamp("model weights saved in history")
            self.actor.load_state_dict(torch.load(self.model_path, map_location=self.device))
            print_with_timestamp("model weights have been updated")
            self.__weights = None
        self.__weights_lock.release()  # END WEIGHTS LOCK...............................................................
