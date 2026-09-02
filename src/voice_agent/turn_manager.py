"""Turn management: IDLE → LISTENING → WAITING_FOR_ENDPOINT → LLM_GENERATING"""
from __future__ import annotations
import time
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Callable

from .nemo_client import ASRPartial

class State(Enum):
    IDLE = auto()
    LISTENING = auto()
    WAITING_FOR_ENDPOINT = auto()
    LLM_GENERATING = auto()

@dataclass
class Turn:
    id: str
    language: str
    start_t: float
    partials: list = field(default_factory=list)
    final_text: Optional[str] = None
    endpoint_t: Optional[float] = None
    asr_final_t: Optional[float] = None
    llm_first_token_t: Optional[float] = None
    llm_done_t: Optional[float] = None
    llm_text: str = ""

class TurnManager:
    def __init__(self, on_commit: Optional[Callable[[str, str], None]] = None):
        self.state = State.IDLE
        self.current: Optional[Turn] = None
        self.turn_counter = 0
        self.on_commit = on_commit  # called with (final_text, language) when committed
        self.history: list[Turn] = []
        self._last_partial_text: str = ""

    def start_turn(self, language: str):
        self.turn_counter += 1
        self.current = Turn(id=f"turn_{self.turn_counter:03d}", language=language, start_t=time.monotonic())
        self.state = State.LISTENING
        self._last_partial_text = ""
        return self.current

    def on_partial(self, p: ASRPartial):
        if self.current is None:
            # auto start if IDLE
            self.start_turn(p.language)
        self.current.partials.append((p.timestamp, p.text, p.is_final))
        self._last_partial_text = p.text
        if p.is_final:
            self.current.final_text = p.text
            self.current.asr_final_t = p.timestamp
            if p.is_endpoint:
                self.current.endpoint_t = p.timestamp
                self.commit_turn()
            else:
                self.state = State.WAITING_FOR_ENDPOINT
        else:
            self.state = State.LISTENING

    def on_endpoint(self, t: Optional[float] = None):
        if self.current and self.current.endpoint_t is None:
            self.current.endpoint_t = t or time.monotonic()
        self.commit_turn()

    def commit_turn(self):
        if self.current is None or self.current.final_text is None:
            return
        # Only commit once
        if self.state == State.LLM_GENERATING:
            return
        self.state = State.LLM_GENERATING
        if self.on_commit:
            # on_commit will trigger LLM and then call complete_llm
            self.on_commit(self.current.final_text, self.current.language)

    def complete_llm(self, llm_text: str, first_token_t: Optional[float] = None, done_t: Optional[float] = None):
        if self.current:
            self.current.llm_text = llm_text
            self.current.llm_first_token_t = first_token_t
            self.current.llm_done_t = done_t or time.monotonic()
            self.history.append(self.current)
        self.current = None
        self.state = State.IDLE

    def handle_barge_in(self, p: ASRPartial):
        """New speech while LLM_GENERATING. For v0, buffer until generation completes."""
        # Log and keep; do not crash. We buffer as new turn that will be committed after.
        # For now, just note: we will start a new turn after current completes.
        # Simple implementation: queue as next turn
        print(f"[turn] barge-in detected while LLM_GENERATING: '{p.text[:40]}' - buffering")
        # we could cancel LLM; for v0 we just buffer
        # Start new turn that will be committed after
        # But we remain in LLM_GENERATING until complete_llm is called
        pass

    def is_listening(self) -> bool:
        return self.state in (State.LISTENING, State.WAITING_FOR_ENDPOINT)

    def latency_breakdown(self) -> dict:
        if not self.history:
            return {}
        last = self.history[-1]
        # Need reference_speech_end - not available here, computed in session
        return {
            "turn": last.id,
            "partials": len(last.partials),
            "endpoint_t": last.endpoint_t,
            "asr_final_t": last.asr_final_t,
            "llm_first_token_t": last.llm_first_token_t,
        }
