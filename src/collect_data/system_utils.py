# src/collect_data/system_utils.py
import gc
import logging
import signal
from datetime import datetime

import psutil
import torch

logger = logging.getLogger(__name__)


def get_memory_usage():
    """Get current memory usage of the process"""
    process = psutil.Process()
    return {
        "ram_percent": process.memory_percent(),
        "ram_used_gb": process.memory_info().rss / (1024 * 1024 * 1024),
        "gpu_memory_used": torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024) if torch.cuda.is_available() else 0,
    }


def clear_memory():
    """Clear GPU and system memory"""
    torch.cuda.empty_cache()
    gc.collect()


def setup_signal_handlers(callback):
    """Set up signal handlers for graceful termination"""

    def signal_handler(signum, frame):
        signal_name = signal.Signals(signum).name
        logger.warning(f"\nReceived signal {signal_name}")
        callback(f"Received signal {signal_name}")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    return signal_handler
