"""Backward-compatible imports for the visualizer's split responsibilities.

New code should import from ``store``, ``log_reader``, or ``presentation`` directly.
"""

from sts2_training.visualizer.log_reader import JsonlLogReader, ReplayLogError, read_jsonl
from sts2_training.visualizer.presentation import present_event
from sts2_training.visualizer.store import EventStore

__all__ = [
    "EventStore",
    "JsonlLogReader",
    "ReplayLogError",
    "present_event",
    "read_jsonl",
]
