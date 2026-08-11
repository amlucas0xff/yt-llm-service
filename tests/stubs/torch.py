"""Stand-in for `torch`, shadowing the real package during tests.

`conftest.py` puts this directory first on `sys.path`. `transcription_service`
imports torch at module scope purely to report device info, and the unit suite
mocks every actual transcription — so pulling in the real ~2GB CUDA wheel just
to answer `cuda.is_available()` would make the tests slow to install and
impossible to run on a machine without a GPU stack.

If a test ever needs real tensors, it belongs in the smoke test
(`scripts/smoke.sh`), which runs against the real container.
"""


class _Cuda:
    @staticmethod
    def is_available():
        return False

    @staticmethod
    def device_count():
        return 0

    @staticmethod
    def current_device():
        return None

    @staticmethod
    def get_device_name():
        return None

    @staticmethod
    def memory_allocated():
        return 0

    @staticmethod
    def memory_reserved():
        return 0


class _Version:
    cuda = None


cuda = _Cuda()
version = _Version()
__version__ = "0.0-test"
