"""Observability and Tracing for Friday."""

import json
import logging
import uuid
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class TraceEvent(BaseModel):
    """A single event in an agent trace."""
    event_type: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    data: Dict[str, Any] = Field(default_factory=dict)
    message: Optional[str] = None

class AgentTrace(BaseModel):
    """A collection of events representing a single agent interaction."""
    trace_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    session_id: Optional[str] = None
    input_text: str
    start_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    events: List[TraceEvent] = Field(default_factory=list)
    final_output: Optional[str] = None
    success: bool = True

class TraceManager:
    """Manages recording and retrieving agent traces."""

    def __init__(self, trace_dir: Optional[Path] = None):
        self.trace_dir = trace_dir or Path.home() / ".friday" / "traces"
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.current_trace: Optional[AgentTrace] = None

    def start_trace(self, input_text: str, session_id: Optional[str] = None) -> AgentTrace:
        """Start a new trace for an interaction."""
        self.current_trace = AgentTrace(input_text=input_text, session_id=session_id)
        self.add_event("input_received", {"text": input_text})
        return self.current_trace

    def add_event(self, event_type: str, data: Optional[Dict[str, Any]] = None, message: Optional[str] = None) -> None:
        """Add an event to the current trace."""
        if not self.current_trace:
            return
        
        event = TraceEvent(event_type=event_type, data=data or {}, message=message)
        self.current_trace.events.append(event)
        logger.debug(f"Trace {self.current_trace.trace_id} event: {event_type} - {message or ''}")

    def end_trace(self, final_output: str, success: bool = True) -> Optional[AgentTrace]:
        """Finalize and save the current trace."""
        if not self.current_trace:
            return None
        
        self.current_trace.end_time = datetime.now(timezone.utc)
        self.current_trace.final_output = final_output
        self.current_trace.success = success
        
        self.save_trace(self.current_trace)
        trace = self.current_trace
        self.current_trace = None
        return trace

    def save_trace(self, trace: AgentTrace) -> None:
        """Save a trace to a JSONL file."""
        try:
            date_str = trace.start_time.strftime("%Y-%m-%d")
            trace_file = self.trace_dir / f"traces-{date_str}.jsonl"
            
            # Convert datetime to string for JSON serialization
            trace_data = trace.model_dump()
            trace_data["start_time"] = trace.start_time.isoformat()
            if trace.end_time:
                trace_data["end_time"] = trace.end_time.isoformat()
            for event in trace_data["events"]:
                event["timestamp"] = event["timestamp"].isoformat()

            with open(trace_file, "a") as f:
                f.write(json.dumps(trace_data) + "\n")
        except Exception as e:
            logger.error(f"Failed to save trace: {e}")

    def get_recent_traces(self, limit: int = 10) -> List[AgentTrace]:
        """Retrieve recent traces from disk."""
        traces = []
        try:
            # Sort trace files by date descending
            files = sorted(self.trace_dir.glob("traces-*.jsonl"), reverse=True)
            for file in files:
                with open(file, "r") as f:
                    # Read lines from end of file would be better, but for small files this is ok
                    lines = f.readlines()
                    for line in reversed(lines):
                        if len(traces) >= limit:
                            break
                        try:
                            data = json.loads(line)
                            # Convert back to AgentTrace
                            # (Simple bypass for complex datetime parsing in pydantic here)
                            traces.append(AgentTrace.model_validate(data))
                        except Exception:
                            continue
                if len(traces) >= limit:
                    break
        except Exception as e:
            logger.error(f"Failed to load traces: {e}")
        return traces

    def get_trace_by_id(self, trace_id: str) -> Optional[AgentTrace]:
        """Find a specific trace by its ID."""
        try:
            files = sorted(self.trace_dir.glob("traces-*.jsonl"), reverse=True)
            for file in files:
                with open(file, "r") as f:
                    for line in f:
                        if trace_id in line:
                            data = json.loads(line)
                            if data.get("trace_id") == trace_id:
                                return AgentTrace.model_validate(data)
        except Exception as e:
            logger.error(f"Failed to find trace {trace_id}: {e}")
        return None

# Global instance
trace_manager = TraceManager()
