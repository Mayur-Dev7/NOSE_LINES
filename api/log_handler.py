"""
Pipeline Log Handler & In-Memory Telemetry Buffer
=================================================
Captures real-time logs from pipeline components and buffers them
for live UI telemetry streaming.
"""

import collections
from datetime import datetime
import logging

# Live in-memory telemetry log buffer for UI streaming
_PIPELINE_LOGS = collections.deque(maxlen=300)


class PipelineLogHandler(logging.Handler):
    """Logging handler that captures records into the _PIPELINE_LOGS deque."""

    def emit(self, record):
        try:
            _PIPELINE_LOGS.append({
                "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
                "level": record.levelname,
                "message": record.getMessage(),
                "logger": record.name,
            })
        except Exception:
            pass


_pipeline_handler = PipelineLogHandler()
_pipeline_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logging.getLogger().addHandler(_pipeline_handler)
logging.getLogger("video_groove_extractor").addHandler(_pipeline_handler)
logging.getLogger("muzzle_cropper").addHandler(_pipeline_handler)
logging.getLogger("dl_muzzle_embedder").addHandler(_pipeline_handler)
logging.getLogger("muzzle_matcher").addHandler(_pipeline_handler)
logging.getLogger("server").addHandler(_pipeline_handler)

logger = logging.getLogger("server")
