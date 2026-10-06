"""Bound actual Ray integration tests to two local workers."""
import os
import shutil
import tempfile
import pytest


@pytest.fixture(autouse=True)
def bounded_ray_runtime(request, tmp_path):
    if request.node.get_closest_marker("ray") is None:
        yield
        return
    import ray
    runtime_dir = tempfile.mkdtemp(prefix="sdb_ray_", dir="/tmp")
    try:
        ray.init(num_cpus=2, object_store_memory=80 * 1024 * 1024,
                 include_dashboard=False, _temp_dir=runtime_dir)
        def remote_pid():
            return os.getpid()
        worker_pid = ray.get(ray.remote(remote_pid).remote())
        assert worker_pid != os.getpid(), "Ray did not execute in a worker process"
        yield
    finally:
        ray.shutdown()
        shutil.rmtree(runtime_dir, ignore_errors=True)
