import argparse
import csv
import os
import queue
import threading
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import yaml

from olympus.deployment import run as local_run
from olympus.deployment import public_iperf
from olympus.deployment import public_iperf_pool
from olympus.deployment import worker as local_worker
from olympus.deployment_batch import run
from olympus.deployment_batch import server
from olympus.deployment_batch import worker


class _RecordingPolicy:
    def __init__(self):
        self.calls = []

    def infer(self, requests):
        self.calls.append([request["flow_id"] for request in requests])
        return [float(request["flow_id"]) for request in requests]


class _FakeSocket:
    def settimeout(self, _timeout):
        pass

    def connect(self, _path):
        pass

    def close(self):
        pass


class DeploymentBatchTest(unittest.TestCase):
    def test_cli_batch_size_must_be_positive(self):
        self.assertEqual(run._positive_int("1"), 1)
        self.assertEqual(server._positive_int("4"), 4)
        with self.assertRaises(argparse.ArgumentTypeError):
            run._positive_int("0")
        with self.assertRaises(argparse.ArgumentTypeError):
            server._positive_int("-1")

    def test_max_batch_size_one_dispatches_singletons(self):
        policy = _RecordingPolicy()
        inference = server.InferenceServer(
            policy, "/tmp/unused-olympus-test.sock",
            max_batch_size=1, max_wait_us=1_000_000)
        done_a = queue.Queue(maxsize=1)
        done_b = queue.Queue(maxsize=1)
        inference.requests.put(({"flow_id": "1", "seq": 1}, done_a))
        inference.requests.put(({"flow_id": "2", "seq": 1}, done_b))
        thread = threading.Thread(target=inference._batch_loop, daemon=True)
        thread.start()
        self.assertTrue(done_a.get(timeout=1)["ok"])
        self.assertTrue(done_b.get(timeout=1)["ok"])
        inference.stop.set()
        thread.join(timeout=1)
        self.assertEqual(policy.calls, [["1"], ["2"]])

    def test_worker_trace_contains_normalized_state_sent_to_server(self):
        state_module = mock.Mock()
        state_module.STATE_DIM = 2
        state_module.normalize_state.return_value = np.asarray(
            [0.25, -0.5], dtype=np.float32)
        action_module = mock.Mock()
        action_module.apply_cwnd.return_value = 10
        raw = {
            "cwnd": 10, "avg_thr": 1_000_000, "avg_urtt": 20_000,
            "srtt_us": 160_000, "min_rtt": 20_000,
        }

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            config = directory / "config.yaml"
            config.write_text(yaml.safe_dump({
                "runtime": {
                    "algorithm": "ma_dreamer",
                    "state": "fake",
                    "action": "cwnd_multiplier",
                },
                "deployment": {"workers": {"interval_ms": 20}},
            }))
            trace_dir = directory / "traces"
            environ = {
                "OLYMPUS_CONFIG": str(config),
                "OLYMPUS_FLOW_FD": "7",
                "OLYMPUS_FLOW_ID": "42",
                "OLYMPUS_INFERENCE_SOCKET": str(directory / "service.sock"),
                "OLYMPUS_TRACE_DIR": str(trace_dir),
            }
            finished = worker.flow_backend.SimulationFinished()
            with mock.patch.dict(os.environ, environ), \
                    mock.patch.object(worker, "load_state_module",
                                      return_value=state_module), \
                    mock.patch.object(worker, "load_action_module",
                                      return_value=action_module), \
                    mock.patch.object(worker.socket, "socket",
                                      return_value=_FakeSocket()), \
                    mock.patch.object(worker, "send_message") as send, \
                    mock.patch.object(worker, "recv_message",
                                      return_value={
                                          "ok": True, "seq": 1,
                                          "action": 0.0}), \
                    mock.patch.object(
                        worker.flow_backend, "get_tcp_deepcc_info",
                        side_effect=[raw, finished]), \
                    mock.patch.object(worker.flow_backend, "set_cwnd"), \
                    mock.patch.object(worker, "sleep_to_grid",
                                      side_effect=lambda tick, _interval: tick):
                worker.run()

            request = send.call_args_list[0].args[1]
            self.assertEqual(request["state"], [0.25, -0.5])
            trace = next(trace_dir.glob("flow_42_pid_*.csv"))
            with trace.open(newline="") as handle:
                reader = csv.DictReader(handle)
                row = next(reader)
                self.assertEqual(reader.fieldnames[-2:], ["s0", "s1"])
                self.assertEqual([float(row["s0"]), float(row["s1"])],
                                 [0.25, -0.5])

    def test_worker_holds_drained_metrics_before_normalization(self):
        state_module = mock.Mock()
        state_module.STATE_DIM = 4
        state_module.normalize_state.side_effect = lambda raw: np.asarray(
            [raw["avg_thr"], raw["avg_urtt"], raw["count"],
             raw["thr_cnt"]], dtype=np.float32)
        action_module = mock.Mock()
        action_module.apply_cwnd.return_value = 10
        first = {
            "cwnd": 10, "avg_thr": 1_000_000, "avg_urtt": 20_000,
            "count": 4, "thr_cnt": 3, "srtt_us": 160_000,
            "min_rtt": 20_000,
        }
        drained = {
            "cwnd": 10, "avg_thr": 0, "avg_urtt": 0,
            "count": 0, "thr_cnt": 0, "srtt_us": 160_000,
            "min_rtt": 20_000,
        }

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            config = directory / "config.yaml"
            config.write_text(yaml.safe_dump({
                "runtime": {
                    "algorithm": "ma_dreamer",
                    "state": "fake",
                    "action": "cwnd_multiplier",
                },
                "deployment": {"workers": {"interval_ms": 20}},
            }))
            trace_dir = directory / "traces"
            environ = {
                "OLYMPUS_CONFIG": str(config),
                "OLYMPUS_FLOW_FD": "7",
                "OLYMPUS_FLOW_ID": "43",
                "OLYMPUS_INFERENCE_SOCKET": str(directory / "service.sock"),
                "OLYMPUS_TRACE_DIR": str(trace_dir),
            }
            finished = worker.flow_backend.SimulationFinished()
            with mock.patch.dict(os.environ, environ), \
                    mock.patch.object(worker, "load_state_module",
                                      return_value=state_module), \
                    mock.patch.object(worker, "load_action_module",
                                      return_value=action_module), \
                    mock.patch.object(worker.socket, "socket",
                                      return_value=_FakeSocket()), \
                    mock.patch.object(worker, "send_message") as send, \
                    mock.patch.object(worker, "recv_message", side_effect=[
                        {"ok": True, "seq": 1, "action": 0.0},
                        {"ok": True, "seq": 2, "action": 0.0},
                    ]), \
                    mock.patch.object(
                        worker.flow_backend, "get_tcp_deepcc_info",
                        side_effect=[first, drained, finished]), \
                    mock.patch.object(worker.flow_backend, "set_cwnd"), \
                    mock.patch.object(worker, "sleep_to_grid",
                                      side_effect=lambda tick, _interval: tick):
                worker.run()

            requests = [call.args[1] for call in send.call_args_list
                        if call.args[1].get("type") == "infer"]
            self.assertEqual(len(requests), 2)
            self.assertEqual(requests[0]["state"],
                             [1_000_000.0, 20_000.0, 4.0, 3.0])
            self.assertEqual(requests[1]["state"], requests[0]["state"])

            trace = next(trace_dir.glob("flow_43_pid_*.csv"))
            with trace.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            second = rows[1]
            self.assertEqual(float(second["avg_thr_bps"]), 1_000_000.0)
            self.assertEqual(float(second["avg_urtt_us"]), 20_000.0)
            self.assertEqual(float(second["sampled_avg_thr_bps"]), 0.0)
            self.assertEqual(float(second["sampled_avg_urtt_us"]), 0.0)
            self.assertEqual(int(second["held_avg_thr"]), 1)
            self.assertEqual(int(second["held_avg_urtt"]), 1)
            self.assertEqual(int(second["held_cnt"]), 1)
            self.assertEqual(int(second["held_thr_cnt"]), 1)


class DeploymentPerFlowTest(unittest.TestCase):
    def test_deployment_modes_are_mutually_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            lock_path = str(Path(directory) / "service.lock")
            handle = local_run._acquire_service_lock(lock_path)
            try:
                with self.assertRaises(SystemExit):
                    run._acquire_service_lock(lock_path)
            finally:
                handle.close()

    def test_worker_owns_model_and_infers_without_batch_server(self):
        state_module = mock.Mock()
        state_module.STATE_DIM = 2
        state_module.normalize_state.return_value = np.asarray(
            [0.25, -0.5], dtype=np.float32)
        action_module = mock.Mock()
        action_module.apply_cwnd.return_value = 11
        policy = mock.Mock()
        policy.infer.return_value = 0.1
        raw = {
            "cwnd": 10, "avg_thr": 1_000_000, "avg_urtt": 20_000,
            "cnt": 4, "thr_cnt": 3, "srtt_us": 160_000,
            "min_rtt": 20_000,
        }

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            config = directory / "config.yaml"
            config.write_text(yaml.safe_dump({
                "runtime": {
                    "algorithm": "ma_dreamer",
                    "state": "fake",
                    "action": "cwnd_multiplier",
                },
                "agent": {"interval_ms": 20, "cwnd_min": 4,
                          "cwnd_max": 10000},
            }))
            environ = {
                "OLYMPUS_CONFIG": str(config),
                "OLYMPUS_CHECKPOINT": str(directory / "model.pt"),
                "OLYMPUS_FLOW_FD": "7",
                "OLYMPUS_FLOW_ID": "99",
            }
            finished = local_worker.flow_backend.SimulationFinished()
            with mock.patch.dict(os.environ, environ), \
                    mock.patch.object(local_worker, "load_state_module",
                                      return_value=state_module), \
                    mock.patch.object(local_worker, "load_action_module",
                                      return_value=action_module), \
                    mock.patch.object(local_worker, "FlowPolicy",
                                      return_value=policy) as policy_factory, \
                    mock.patch.object(
                        local_worker.flow_backend, "get_tcp_deepcc_info",
                        side_effect=[raw, finished]), \
                    mock.patch.object(
                        local_worker.flow_backend, "set_cwnd") as set_cwnd, \
                    mock.patch.object(local_worker, "sleep_to_grid",
                                      side_effect=lambda tick, _interval: tick):
                local_worker.run()

            policy_factory.assert_called_once()
            policy.infer.assert_called_once_with([0.25, -0.5])
            set_cwnd.assert_called_once_with(7, 11)


    def _run_orca_warmup_worker(self, samples, warmup_s):
        """Drive the per-flow worker over `samples`, returning its set_cwnd calls."""
        state_module = mock.Mock()
        state_module.STATE_DIM = 2
        state_module.normalize_state.return_value = np.asarray(
            [0.25, -0.5], dtype=np.float32)
        action_module = mock.Mock()
        action_module.apply_cwnd.return_value = 11
        policy = mock.Mock()
        policy.infer.return_value = 0.1

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            config = directory / "config.yaml"
            config.write_text(yaml.safe_dump({
                "runtime": {"algorithm": "orca", "state": "fake",
                            "action": "cwnd_multiplier"},
                "agent": {"interval_ms": 20, "cwnd_min": 4, "cwnd_max": 10000,
                          "cubic_warmup": True,
                          "cubic_warmup_max_s": warmup_s},
            }))
            environ = {
                "OLYMPUS_CONFIG": str(config),
                "OLYMPUS_CHECKPOINT": str(directory / "model.pt"),
                "OLYMPUS_FLOW_FD": "7",
                "OLYMPUS_FLOW_ID": "99",
            }
            finished = local_worker.flow_backend.SimulationFinished()
            # Each poll advances the clock by half the warmup window.
            clock = [0.0]

            def tick():
                clock[0] += warmup_s / 2.0
                return clock[0]

            with mock.patch.dict(os.environ, environ), \
                    mock.patch.object(local_worker, "load_state_module",
                                      return_value=state_module), \
                    mock.patch.object(local_worker, "load_action_module",
                                      return_value=action_module), \
                    mock.patch.object(local_worker, "FlowPolicy",
                                      return_value=policy), \
                    mock.patch.object(
                        local_worker.flow_backend, "get_tcp_deepcc_info",
                        side_effect=list(samples) + [finished]), \
                    mock.patch.object(
                        local_worker.flow_backend, "set_cwnd") as set_cwnd, \
                    mock.patch.object(local_worker.time, "monotonic",
                                      side_effect=tick), \
                    mock.patch.object(local_worker, "sleep_to_grid",
                                      side_effect=lambda t, _i: t):
                local_worker.run()
        return policy, set_cwnd

    def test_orca_warmup_ignores_loss_and_holds_for_the_full_window(self):
        """Loss inside the window must NOT hand over early."""
        lossy = {"cwnd": 10, "avg_thr": 1_000_000, "avg_urtt": 20_000,
                 "cnt": 4, "thr_cnt": 3, "srtt_us": 160_000,
                 "min_rtt": 20_000, "loss_bytes": 4344}
        policy, set_cwnd = self._run_orca_warmup_worker([dict(lossy)], 1.0)
        policy.infer.assert_not_called()
        set_cwnd.assert_not_called()

    def test_orca_warmup_hands_over_once_the_window_elapses(self):
        raw = {"cwnd": 10, "avg_thr": 1_000_000, "avg_urtt": 20_000,
               "cnt": 4, "thr_cnt": 3, "srtt_us": 160_000, "min_rtt": 20_000}
        policy, set_cwnd = self._run_orca_warmup_worker(
            [dict(raw), dict(raw), dict(raw)], 1.0)
        # Samples land at t=0.5/1.0/1.5s (the clock also ticks once at setup),
        # so the agent owns the tail of the run but not its start.
        self.assertGreater(policy.infer.call_count, 0)
        self.assertLess(policy.infer.call_count, 3)
        set_cwnd.assert_called_with(7, 11)


class PublicIperfTest(unittest.TestCase):
    def test_parallel_stream_interval_series(self):
        stream = lambda bps: {
            "intervals": [{
                "streams": [{
                    "socket": 5, "end": 1.0,
                    "bits_per_second": bps,
                }],
            }],
        }
        series = public_iperf._interval_series(stream(10_000_000))
        self.assertEqual(series["5"], [[1.0], [10.0]])

    def test_server_pool_schedule_has_common_end_time(self):
        schedule = public_iperf_pool._flow_schedule(
            count=3, duration=40, delay=4, omit=2)
        self.assertEqual(schedule, [(0.0, 38.0), (4.0, 34.0),
                                    (8.0, 30.0)])
        for (start, measured) in schedule:
            self.assertEqual(start + 2 + measured, 40.0)

    def test_server_pool_merge_offsets_and_labels_independent_flows(self):
        entries = [{
            "index": 2,
            "server": "iperf.example",
            "data_offset_s": 6.0,
            "payload": {
                "intervals": [{
                    "streams": [{
                        "socket": 5,
                        "start": 0.0,
                        "end": 1.0,
                        "bits_per_second": 10_000_000,
                    }],
                }],
            },
        }]
        merged = public_iperf_pool._merge_intervals(entries)
        stream = merged["intervals"][0]["streams"][0]
        self.assertEqual(stream["socket"], "2:iperf.example")
        self.assertEqual(stream["start"], 6.0)
        self.assertEqual(stream["end"], 7.0)


if __name__ == "__main__":
    unittest.main()
