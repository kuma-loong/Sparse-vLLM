import os
import threading
import time
from multiprocessing import get_context
from multiprocessing.shared_memory import SharedMemory
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from sparsevllm.engine.model_runner import ModelRunner


def test_write_shm_waits_until_worker_reads_command():
    ctx = get_context("spawn")
    event = ctx.Event()
    shm = SharedMemory(
        name=f"sparsevllm_test_rpc_{os.getpid()}_{uuid4().hex}",
        create=True,
        size=2**20,
    )
    rank0 = SimpleNamespace(world_size=2, rank=0, event=[event], shm=shm)
    worker = SimpleNamespace(world_size=2, rank=1, event=event, shm=shm)
    errors: list[BaseException] = []

    def write_command():
        try:
            ModelRunner.write_shm(rank0, "free_slots", 123)
        except BaseException as exc:  # pragma: no cover - surfaced below.
            errors.append(exc)

    writer = threading.Thread(target=write_command)
    writer.start()

    try:
        assert event.wait(timeout=1.0)
        time.sleep(0.02)
        assert writer.is_alive()

        method_name, args = ModelRunner.read_shm(worker)
        writer.join(timeout=1.0)

        assert not writer.is_alive()
        assert errors == []
        assert method_name == "free_slots"
        assert args == [123]
    finally:
        if event.is_set():
            event.clear()
        writer.join(timeout=1.0)
        shm.close()
        shm.unlink()


def test_exit_rpc_does_not_wait_for_worker_ack():
    ctx = get_context("spawn")
    event = ctx.Event()
    shm = SharedMemory(
        name=f"sparsevllm_test_exit_rpc_{os.getpid()}_{uuid4().hex}",
        create=True,
        size=2**20,
    )
    exited: list[bool] = []
    runner = object.__new__(ModelRunner)
    runner.world_size = 2
    runner.rank = 0
    runner.event = [event]
    runner.shm = shm
    runner.exit = lambda: exited.append(True)

    try:
        ModelRunner.call(runner, "exit")

        assert event.is_set()
        assert exited == [True]
    finally:
        if event.is_set():
            event.clear()
        shm.close()
        shm.unlink()


def test_model_runner_exit_does_not_use_tp_barrier():
    closed: list[str] = []

    class FakeShm:
        def close(self):
            closed.append("close")

        def unlink(self):
            closed.append("unlink")

    runner = object.__new__(ModelRunner)
    runner.world_size = 2
    runner.rank = 0
    runner.shm = FakeShm()

    with patch("sparsevllm.engine.model_runner.dist.barrier") as barrier, patch(
        "sparsevllm.engine.model_runner.dist.destroy_process_group"
    ) as destroy_process_group, patch("sparsevllm.engine.model_runner.torch.cuda.synchronize") as synchronize:
        ModelRunner.exit(runner)

    barrier.assert_not_called()
    synchronize.assert_called_once_with()
    destroy_process_group.assert_called_once_with()
    assert closed == ["close", "unlink"]


def test_free_slots_batch_releases_each_seq_id():
    freed: list[int] = []

    class FakeCacheManager:
        def free_seq(self, seq_id: int):
            freed.append(int(seq_id))

    runner = object.__new__(ModelRunner)
    runner.cache_manager = FakeCacheManager()

    ModelRunner.free_slots_batch(runner, [3, 5, 8])

    assert freed == [3, 5, 8]
