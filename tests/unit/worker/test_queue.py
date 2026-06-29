import asyncio

from seshat.models.enums import JobStatus
from seshat.worker.queue import AsyncioTaskQueue


class TestAsyncioTaskQueue:
    async def test_enqueue_and_get_status(self):
        queue = AsyncioTaskQueue()

        async def noop():
            pass

        await queue.enqueue("job-1", noop)
        await asyncio.sleep(0)
        status = await queue.get_status("job-1")
        assert status == JobStatus.DONE

    async def test_get_status_unknown_job(self):
        queue = AsyncioTaskQueue()
        status = await queue.get_status("nonexistent-job")
        assert status is None

    async def test_cancel_pending_job(self):
        queue = AsyncioTaskQueue()
        gate = asyncio.Event()

        async def slow():
            await gate.wait()

        await queue.enqueue("job-2", slow)
        cancelled = await queue.cancel("job-2")
        gate.set()
        assert cancelled is True
