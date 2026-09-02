from voice_agent.turn_manager import TurnManager
from voice_agent.nemo_client import ASRPartial
import time

def test_turn():
    commits=[]
    tm=TurnManager(on_commit=lambda t,l: commits.append(t))
    tm.start_turn("en-US")
    tm.on_partial(ASRPartial("hello", time.monotonic(), False))
    assert tm.state.name=="LISTENING"
    tm.on_partial(ASRPartial("hello world", time.monotonic(), True, True))
    assert tm.state.name=="LLM_GENERATING"
    assert commits[0]=="hello world"
    tm.complete_llm("hi there", time.monotonic(), time.monotonic())
    assert tm.state.name=="IDLE"
    print("ok")

if __name__=="__main__":
    test_turn()
