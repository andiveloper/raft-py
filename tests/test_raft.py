"""
Tests that check whether the Raft implementation behaves like Raft.

Two groups of tests:

* Unit tests that feed messages directly into the state objects and the
  serializer (no network involved).
* Integration tests that start a real cluster of RaftServer instances on the
  loopback interface (127.0.0.1, 127.0.0.2, ...) and check the guarantees of
  the Raft paper: a single leader gets elected, all nodes agree on it, the
  leader stays stable, client requests are redirected/accepted, log entries
  get replicated and committed, and a new leader takes over when the old one
  dies.

The implementation is written for Python 2, run from the repository root:

    python2 -m unittest discover -s tests -v

Set RAFT_TEST_VERBOSE=1 to see the log output of the servers.
"""
import logging
import os
import sys
import threading
import time
import unittest
import zlib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from raft_config import RaftConfig  # noqa: E402
from raft_server import RaftServer  # noqa: E402
from persistence.synchronized_log import SynchronizedLog  # noqa: E402
from rpc.messages import (AppendEntries, AppendEntriesResponse, ClientData, ClientDataResponse,  # noqa: E402
                          LogEntry, RequestVote, RequestVoteResponse)
from rpc.serializer import RaftSerializer  # noqa: E402
from states.states import Candidate, Follower, Leader  # noqa: E402

if not os.environ.get("RAFT_TEST_VERBOSE"):
    logging.getLogger().setLevel(logging.CRITICAL)

# Upper bound for an election: max election timeout plus generous slack for
# the 1s retry loop in RPCHandler.send and thread scheduling.
ELECTION_DEADLINE_SECONDS = (RaftConfig.ELECTION_TIMEOUT_IN_MILLIS_MAX / 1000.0) * 3 + 5
REPLICATION_DEADLINE_SECONDS = 5


def wait_until(predicate, timeout, interval=0.05):
    """Poll predicate until it returns a truthy value or timeout expires."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(interval)
    return predicate()


# --------------------------------------------------------------------------
# Unit tests
# --------------------------------------------------------------------------

class FakeServer(object):
    """Minimal stand-in for RaftServer so states can be tested without sockets."""

    def __init__(self, hostname="node-a", peers=("node-b", "node-c")):
        self.hostname = hostname
        self.peers = list(peers)
        self.shutdown = True  # keeps Leader heartbeat / Candidate threads from looping
        self.state = None
        self.broadcasted = []

    def broadcast(self, obj):
        self.broadcasted.append(obj)


def make_follower(term=0, voted_for=None, hostname="node-a"):
    server = FakeServer(hostname)
    server.state = Follower(server, term, voted_for, SynchronizedLog(), None)
    return server


class SerializerTest(unittest.TestCase):
    """Every message must survive the serialize -> compress -> decompress -> deserialize trip."""

    def roundtrip(self, obj):
        serializer = RaftSerializer()
        data = zlib.compress(serializer.serialize(obj))
        return serializer.deserialize(zlib.decompress(data))

    def test_request_vote(self):
        obj = self.roundtrip(RequestVote(3, "127.0.0.2", 7, 2))
        self.assertIsInstance(obj, RequestVote)
        self.assertEqual((obj.term, obj.candidateId, obj.lastLogIndex, obj.lastLogTerm), (3, "127.0.0.2", 7, 2))

    def test_request_vote_response(self):
        for granted in (True, False):
            obj = self.roundtrip(RequestVoteResponse(5, granted))
            self.assertIsInstance(obj, RequestVoteResponse)
            self.assertEqual((obj.term, obj.voteGranted), (5, granted))

    def test_append_entries_without_entries(self):
        obj = self.roundtrip(AppendEntries(4, "127.0.0.1", 2, 1, 2, []))
        self.assertIsInstance(obj, AppendEntries)
        self.assertEqual((obj.term, obj.leaderId, obj.prevLogIndex, obj.prevLogTerm, obj.leaderCommitIndex),
                         (4, "127.0.0.1", 2, 1, 2))
        self.assertEqual(obj.logEntries, [])

    def test_append_entries_with_entries(self):
        entries = [LogEntry(1, 1, "hello"), LogEntry(2, 1, "with:separators;and,commas")]
        obj = self.roundtrip(AppendEntries(3, "127.0.0.1", 4, 2, 4, entries))
        self.assertEqual((obj.term, obj.leaderId, obj.prevLogIndex, obj.prevLogTerm, obj.leaderCommitIndex),
                         (3, "127.0.0.1", 4, 2, 4))
        self.assertEqual([(e.index, e.term, e.data) for e in obj.logEntries],
                         [(1, 1, "hello"), (2, 1, "with:separators;and,commas")])

    def test_append_entries_response(self):
        for success in (True, False):
            obj = self.roundtrip(AppendEntriesResponse(6, success))
            self.assertIsInstance(obj, AppendEntriesResponse)
            self.assertEqual((obj.term, obj.success), (6, success))

    def test_client_data(self):
        obj = self.roundtrip(ClientData("hello world"))
        self.assertIsInstance(obj, ClientData)
        self.assertEqual(obj.data, "hello world")

    def test_client_data_response(self):
        obj = self.roundtrip(ClientDataResponse(True, "127.0.0.3"))
        self.assertIsInstance(obj, ClientDataResponse)
        self.assertEqual((obj.success, obj.leaderId), (True, "127.0.0.3"))


class RequestVoteRulesTest(unittest.TestCase):
    """Raft paper, figure 2, RequestVote RPC receiver implementation."""

    def test_grants_vote_to_first_candidate_in_term(self):
        server = make_follower(term=0)
        resp = server.state.handle(RequestVote(1, "node-b", 0, 0))
        self.assertTrue(resp.voteGranted)
        self.assertEqual(resp.term, 1)
        self.assertEqual(server.state.votedFor, "node-b")

    def test_grants_vote_again_to_same_candidate(self):
        server = make_follower(term=0)
        server.state.handle(RequestVote(1, "node-b", 0, 0))
        resp = server.state.handle(RequestVote(1, "node-b", 0, 0))
        self.assertTrue(resp.voteGranted)

    def test_only_one_vote_per_term(self):
        server = make_follower(term=0)
        self.assertTrue(server.state.handle(RequestVote(1, "node-b", 0, 0)).voteGranted)
        resp = server.state.handle(RequestVote(1, "node-c", 0, 0))
        self.assertFalse(resp.voteGranted, "a node must not vote for two candidates in the same term")

    def test_rejects_candidate_with_stale_term(self):
        server = make_follower(term=5)
        resp = server.state.handle(RequestVote(4, "node-b", 0, 0))
        self.assertFalse(resp.voteGranted)
        self.assertEqual(resp.term, 5)

    def test_grants_vote_in_new_term_after_voting_in_old_term(self):
        server = make_follower(term=0)
        self.assertTrue(server.state.handle(RequestVote(1, "node-b", 0, 0)).voteGranted)
        resp = server.state.handle(RequestVote(2, "node-c", 0, 0))
        self.assertTrue(resp.voteGranted, "votedFor must be reset when a higher term is seen")
        self.assertEqual(server.state.currentTerm, 2)

    def test_rejects_candidate_with_less_up_to_date_log(self):
        server = make_follower(term=1)
        server.state.log.append_entries([LogEntry(1, 1, "a"), LogEntry(2, 1, "b")])
        server.state.log.commitIndex = 2
        resp = server.state.handle(RequestVote(2, "node-b", 0, 0))
        self.assertFalse(resp.voteGranted, "candidate's log is behind ours, vote must be denied")


class AppendEntriesRulesTest(unittest.TestCase):
    """Raft paper, figure 2, AppendEntries RPC receiver implementation."""

    def test_rejects_stale_leader(self):
        server = make_follower(term=3)
        resp = server.state.handle(AppendEntries(2, "node-b", 0, 0, 0, []))
        self.assertFalse(resp.success)
        self.assertEqual(resp.term, 3)

    def test_heartbeat_sets_leader_and_term(self):
        server = make_follower(term=0)
        resp = server.state.handle(AppendEntries(2, "node-b", 0, 0, 0, []))
        self.assertTrue(resp.success)
        self.assertEqual(server.state.currentTerm, 2)
        self.assertEqual(server.state.currentLeaderId, "node-b")

    def test_appends_entries(self):
        server = make_follower(term=1)
        resp = server.state.handle(AppendEntries(1, "node-b", 0, 0, 0, [LogEntry(1, 1, "x")]))
        self.assertTrue(resp.success)
        self.assertIn(1, server.state.log.logEntries)
        self.assertEqual(server.state.log.logEntries[1].data, "x")

    def test_consecutive_appends(self):
        server = make_follower(term=1)
        self.assertTrue(server.state.handle(AppendEntries(1, "node-b", 0, 0, 0, [LogEntry(1, 1, "x")])).success)
        resp = server.state.handle(AppendEntries(1, "node-b", 1, 1, 0, [LogEntry(2, 1, "y")]))
        self.assertTrue(resp.success, "entry at prevLogIndex/prevLogTerm exists, append must succeed")
        self.assertEqual(sorted(server.state.log.logEntries), [1, 2])

    def test_rejects_when_prev_log_entry_is_missing(self):
        server = make_follower(term=1)
        resp = server.state.handle(AppendEntries(1, "node-b", 5, 1, 0, [LogEntry(6, 1, "x")]))
        self.assertFalse(resp.success, "log has no entry at prevLogIndex, append must fail")

    def test_advances_commit_index(self):
        server = make_follower(term=1)
        server.state.handle(AppendEntries(1, "node-b", 0, 0, 0, [LogEntry(1, 1, "x"), LogEntry(2, 1, "y")]))
        server.state.handle(AppendEntries(1, "node-b", 2, 1, 2, []))
        self.assertEqual(server.state.log.commitIndex, 2)

    def test_candidate_steps_down_on_append_entries_of_same_term(self):
        server = make_follower(term=0)
        server.state = server.state.switch_to(Candidate)
        server.state._start_election_thread.join()
        term = server.state.currentTerm
        server.state.handle(AppendEntries(term, "node-b", 0, 0, 0, []))
        self.assertIsInstance(server.state, Follower)
        self.assertEqual(server.state.currentLeaderId, "node-b")


class CandidateTest(unittest.TestCase):

    def test_candidate_increments_term_votes_for_itself_and_requests_votes(self):
        server = make_follower(term=4)
        server.state = server.state.switch_to(Candidate)
        server.state._start_election_thread.join()
        self.assertEqual(server.state.currentTerm, 5)
        self.assertEqual(server.state.votedFor, "node-a")
        self.assertEqual(len(server.broadcasted), 1)
        self.assertIsInstance(server.broadcasted[0], RequestVote)
        self.assertEqual(server.broadcasted[0].term, 5)

    def test_candidate_becomes_leader_with_majority(self):
        server = make_follower(term=0)
        server.state = server.state.switch_to(Candidate)
        server.state._start_election_thread.join()
        server.state.handle(RequestVoteResponse(server.state.currentTerm, True))
        self.assertIsInstance(server.state, Leader)
        self.assertEqual(server.state.currentLeaderId, "node-a")

    def test_candidate_starts_election_after_having_voted_in_older_term(self):
        # A follower that voted for someone in term 1 times out and becomes a candidate:
        # it must start a new election for term 2.
        server = make_follower(term=0)
        server.state.handle(RequestVote(1, "node-b", 0, 0))
        server.state = server.state.switch_to(Candidate)
        server.state._start_election_thread.join()
        self.assertEqual(server.state.currentTerm, 2)
        self.assertEqual(len(server.broadcasted), 1, "no RequestVote was sent")


class SingleNodeTest(unittest.TestCase):

    def test_single_node_is_leader_and_commits(self):
        server = FakeServer(peers=())
        server.state = Leader(server, 0, None, SynchronizedLog(), None)
        resp = server.state.handle(ClientData("hello"))
        self.assertTrue(resp.success)
        self.assertEqual(server.state.log.commitIndex, 1)
        self.assertEqual(server.state.log.logEntries[1].data, "hello")


# --------------------------------------------------------------------------
# Integration tests: real servers talking over TCP on the loopback interface
# --------------------------------------------------------------------------

_next_port = [47900 + (os.getpid() % 500) * 10]


class Cluster(object):
    """Starts n RaftServer instances on 127.0.0.1 .. 127.0.0.n.

    Raft uses the hostname as the node id, so every node gets its own loopback
    address (Linux routes all of 127.0.0.0/8 to lo).
    """

    def __init__(self, size):
        port = _next_port[0]
        _next_port[0] += size + 1
        self.addresses = [("127.0.0.%d" % (i + 1), port + i) for i in range(size)]
        self.servers = []
        for address in self.addresses:
            peers = [a for a in self.addresses if a != address]
            self.servers.append(RaftServer(peers, hostname=address[0], port=address[1]))
        self.stopped = set()
        for server in self.servers:
            server.start()

    def alive(self):
        return [s for s in self.servers if s.hostname not in self.stopped]

    def leaders(self, servers=None):
        return [s for s in (servers or self.alive()) if isinstance(s.state, Leader)]

    def stable_leader(self, servers=None):
        """Return the leader if exactly one exists and every alive node agrees on it."""
        servers = servers or self.alive()
        leaders = self.leaders(servers)
        if len(leaders) != 1:
            return None
        leader = leaders[0]
        for s in servers:
            if s.state.currentLeaderId != leader.hostname or s.state.currentTerm != leader.state.currentTerm:
                return None
        return leader

    def wait_for_leader(self, timeout=ELECTION_DEADLINE_SECONDS):
        return wait_until(self.stable_leader, timeout)

    def stop_server(self, server):
        self.stopped.add(server.hostname)
        server.stop()

    def stop(self):
        threads = [threading.Thread(target=s.stop) for s in self.servers if s.hostname not in self.stopped]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.stopped = set(s.hostname for s in self.servers)


def send_client_data(address, data):
    """Send ClientData to a node like the --client mode of raft_server.py does."""
    client = RaftServer([], hostname="127.0.0.250", port=0)
    try:
        return client.send(address[0], address[1], ClientData(data))
    finally:
        client.stop()


class ClusterTestCase(unittest.TestCase):
    CLUSTER_SIZE = 3

    def setUp(self):
        self.cluster = Cluster(self.CLUSTER_SIZE)

    def tearDown(self):
        self.cluster.stop()

    def require_leader(self):
        leader = self.cluster.wait_for_leader()
        self.assertIsNotNone(leader, "no leader agreed on by all nodes within %ss: %s"
                             % (ELECTION_DEADLINE_SECONDS, self.describe()))
        return leader

    def describe(self):
        return ", ".join("%s=%s(term=%s, leader=%s)" % (s.hostname, s.state.__class__.__name__,
                                                       s.state.currentTerm, s.state.currentLeaderId)
                         for s in self.cluster.servers)


class ElectionTest(ClusterTestCase):

    def test_elects_exactly_one_leader(self):
        leader = self.require_leader()
        self.assertGreaterEqual(leader.state.currentTerm, 1)
        followers = [s for s in self.cluster.servers if s is not leader]
        for f in followers:
            self.assertIsInstance(f.state, Follower)

    def test_at_most_one_leader_per_term(self):
        # Election safety: sample the cluster during startup and the following seconds.
        leaders_per_term = {}
        deadline = time.time() + ELECTION_DEADLINE_SECONDS + 2
        while time.time() < deadline:
            for s in self.cluster.servers:
                state = s.state
                if isinstance(state, Leader):
                    leaders_per_term.setdefault(state.currentTerm, set()).add(s.hostname)
            time.sleep(0.02)
        self.assertTrue(leaders_per_term, "no leader was ever elected")
        for term, leaders in leaders_per_term.items():
            self.assertEqual(len(leaders), 1, "term %d had several leaders: %s" % (term, sorted(leaders)))

    def test_leader_stays_stable_with_heartbeats(self):
        leader = self.require_leader()
        term = leader.state.currentTerm
        # Several election timeouts long: heartbeats must keep followers from starting elections.
        time.sleep(RaftConfig.ELECTION_TIMEOUT_IN_MILLIS_MAX / 1000.0 * 2)
        self.assertIs(self.cluster.stable_leader(), leader, self.describe())
        self.assertEqual(leader.state.currentTerm, term, "term changed although the leader was healthy")


class FiveNodeElectionTest(ElectionTest):
    CLUSTER_SIZE = 5


class ClientRequestTest(ClusterTestCase):

    def test_follower_redirects_client_to_leader(self):
        leader = self.require_leader()
        follower = [s for s in self.cluster.servers if s is not leader][0]
        resp = send_client_data((follower.hostname, follower.port), "hello")
        self.assertIsInstance(resp, ClientDataResponse)
        self.assertFalse(resp.success)
        self.assertEqual(resp.leaderId, leader.hostname)

    def test_leader_accepts_client_data(self):
        leader = self.require_leader()
        resp = send_client_data((leader.hostname, leader.port), "hello")
        self.assertIsInstance(resp, ClientDataResponse)
        self.assertTrue(resp.success)
        self.assertEqual(resp.leaderId, leader.hostname)


class ReplicationTest(ClusterTestCase):

    def entries(self, server):
        log = server.state.log
        return [(i, log.logEntries[i].data) for i in sorted(log.logEntries)]

    def test_entries_are_replicated_to_all_followers(self):
        leader = self.require_leader()
        for data in ("first", "second", "third"):
            self.assertTrue(send_client_data((leader.hostname, leader.port), data).success)
        expected = [(1, "first"), (2, "second"), (3, "third")]
        replicated = wait_until(lambda: all(self.entries(s) == expected for s in self.cluster.servers),
                                REPLICATION_DEADLINE_SECONDS)
        self.assertTrue(replicated, "logs differ: %s"
                        % dict((s.hostname, self.entries(s)) for s in self.cluster.servers))

    def test_entries_get_committed_on_all_nodes(self):
        leader = self.require_leader()
        for data in ("first", "second"):
            self.assertTrue(send_client_data((leader.hostname, leader.port), data).success)
        committed = wait_until(lambda: all(s.state.log.commitIndex == 2 for s in self.cluster.servers),
                               REPLICATION_DEADLINE_SECONDS)
        self.assertTrue(committed, "commit indexes: %s"
                        % dict((s.hostname, s.state.log.commitIndex) for s in self.cluster.servers))

    def test_log_entries_carry_the_leaders_term(self):
        leader = self.require_leader()
        self.assertTrue(send_client_data((leader.hostname, leader.port), "x").success)
        self.assertEqual(leader.state.log.logEntries[1].term, leader.state.currentTerm,
                         "new entries must be stored with the leader's current term")


class FailoverTest(ClusterTestCase):

    def test_new_leader_is_elected_after_leader_crash(self):
        old_leader = self.require_leader()
        old_term = old_leader.state.currentTerm
        self.cluster.stop_server(old_leader)

        new_leader = self.cluster.wait_for_leader()
        self.assertIsNotNone(new_leader, "no new leader after the old one was stopped: %s" % self.describe())
        self.assertIsNot(new_leader, old_leader)
        self.assertGreater(new_leader.state.currentTerm, old_term)

    def test_cluster_accepts_client_data_after_failover(self):
        old_leader = self.require_leader()
        self.cluster.stop_server(old_leader)
        new_leader = self.cluster.wait_for_leader()
        self.assertIsNotNone(new_leader, "no new leader after the old one was stopped: %s" % self.describe())
        resp = send_client_data((new_leader.hostname, new_leader.port), "after failover")
        self.assertTrue(resp.success)


if __name__ == "__main__":
    unittest.main()
