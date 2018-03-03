import socket
import threading
from threading import Thread
import argparse
import logging
from time import sleep
import zlib

from rpc.messages import AppendEntries, LogEntry, ClientData, ClientDataResponse
from rpc.rpc_handler import RPCHandler
from rpc.serializer import ScaleRaftSerializer
from scale_raft_config import ScaleRaftConfig
from states.states import Follower, Candidate, Leader

logger = logging.getLogger(__name__)
log_format = '%(asctime)s - %(levelname)s - %(module)s - %(threadName)s - %(message)s'
# logging.basicConfig(format=logFormat, filename=ScaleRaftConfig().LOG_FILE, level=ScaleRaftConfig().LOG_LEVEL)
logging.basicConfig(format=log_format, level=ScaleRaftConfig().LOG_LEVEL)


class ZLibCompressor(object):
    def compress(self, data):
        return zlib.compress(data)

    def decompress(self, data):
        return zlib.decompress(data)


class NoOpEncryptor(object):
    def encrypt(self, data):
        return data

    def decrypt(self, data):
        return data


class ScaleRaftServer(object):
    def __init__(self, peers, hostname=ScaleRaftConfig().HOSTNAME, port=ScaleRaftConfig().PORT,
                 compressor=ZLibCompressor, encryptor=NoOpEncryptor, rpc_handler=RPCHandler,
                 serializer=ScaleRaftSerializer):
        self.peers = peers

        self.hostname = hostname
        self.port = port

        self.__stateLock = threading.RLock()

        self._state = None

        self.__compressor = compressor()
        self.__encryptor = encryptor()

        self.__rpc_handler = rpc_handler(self.hostname, self.port, self._handle_msg, self._handle_timeout)
        self.__serializer = serializer()

        self.__send_threads = []

        self.__stop = False

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        self.__stateLock.acquire()
        logger.info("Switching state from {} to {}".format(self._state.__class__.__name__,
                                                           state.__class__.__name__))
        self._state = state
        self.__stateLock.release()

    def _handle_timeout(self):
        if isinstance(self.state, Follower) or isinstance(self.state, Candidate):
            self.state = Candidate(self)

    def _handle_msg(self, string):
        obj = self._deserialize(string)
        resp_obj = self.state.handle(obj)

        # wait until a new leader is found
        if isinstance(resp_obj, ClientDataResponse):
            if not resp_obj.success and resp_obj.leaderId is None:
                while not self.__stop and self.state.currentLeaderId is None:
                    logger.error("Received client request but currently no leader, wait 1 second...")
                    sleep(1)

        string = None
        if resp_obj is not None:
            string = self._serialize(resp_obj)
        return string

    def _serialize(self, obj):
        if obj is None:
            return None
        serialized_string = self.__serializer.serialize(obj)
        serialized_string = self.__compressor.compress(serialized_string)
        serialized_string = self.__encryptor.encrypt(serialized_string)
        return serialized_string

    def _deserialize(self, string):
        if string is None or string == '':
            return None
        string = self.__encryptor.decrypt(string)
        string = self.__compressor.decompress(string)
        obj = self.__serializer.deserialize(string)
        return obj

    def start(self):
        logger.info("Starting server...")
        logger.info("Peers: {}".format(self.peers))
        if len(self.peers) == 0:
            logger.info("No peers configured, starting as Leader")
            self._state = Leader(self)
        else:
            logger.info("{} peers configured, starting as Follower".format(len(self.peers)))
            self._state = Follower(self)

        self.__rpc_handler.startup()

        logger.info("Server listening on {}:{}".format(self.hostname, self.port))

    def send(self, hostname, port, obj):
        serialized_string = self._serialize(obj)
        return self._deserialize(self.__rpc_handler.send(hostname, port, serialized_string))

    def _send_and_handle(self, hostname, port, obj):
        serialized_string = self._serialize(obj)
        self.state.handle(self._deserialize(self.__rpc_handler.send(hostname, port, serialized_string)))

    def send_and_handle_async(self, hostname, port, obj):
        t = Thread(target=self._send_and_handle, args=(hostname, port, obj))
        self.__send_threads.append(t)
        t.start()

    def broadcast(self, obj):
        # self.send_and_handle_async(self.hostname, ScaleRaftConfig().PORT, obj)
        for peer in self.peers:
            host = peer
            port = int(ScaleRaftConfig().PORT)
            if isinstance(peer, tuple):
                host = peer[0]
                port = int(peer[1])
            self.send_and_handle_async(host, port, obj)

    def stop(self):
        self.__stop = True
        self.__rpc_handler.shutdown()
        for t in self.__send_threads:
            while t.is_alive():
                pass
        logger.info("Server stopped successfully.")


if __name__ == "__main__":
    argparse = argparse.ArgumentParser(description="Start a new server or send a message as a client")
    argparse.add_argument("--server", action="store_true")
    argparse.add_argument("--peers", type=str, default=ScaleRaftConfig().PORT)
    argparse.add_argument("--client", action="store_true")
    argparse.add_argument("--host", type=str, default=ScaleRaftConfig().HOSTNAME)
    argparse.add_argument("--port", type=int, default=ScaleRaftConfig().PORT)
    args = argparse.parse_args()

    # python scale_raft_server.py --server --host andi-vbox --port 48000 --peers localhost:48001,127.0.0.1:48002
    # python scale_raft_server.py --server --host localhost --port 48001 --peers andi-vbox:48000,127.0.0.1:48002
    # python scale_raft_server.py --server --host 127.0.0.1 --port 48002 --peers localhost:48001,andi-vbox:48000
    if args.server:
        peer_tuples = []
        for peer in args.peers.split(","):
            splitted_peer = peer.split(":")
            peer_tuples.append((splitted_peer[0], splitted_peer[1]))

        server = ScaleRaftServer(peer_tuples, hostname=args.host, port=args.port)
        try:
            server.start()
        except Exception as e:
            logger.exception(e)
            server.stop()
            exit(1)

    if args.client:
        try:
            server = ScaleRaftServer([])
            logger.info("Connecting to: {}:{}".format(args.host, args.port))
            resp = server.send(args.host, args.port, ClientData("hello world"))
            print "Success: {}, Leader: {}".format(resp.success, resp.leaderId)
        except Exception as e:
            logger.exception(e)
            exit(1)







