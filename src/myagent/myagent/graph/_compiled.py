from __future__ import annotations

from typing import Any, Optional

from myagent_core.runnable.config import RunnableConfig

from myagent.pregel.loop import Pregel
from myagent.types import StateSnapshot


class CompiledStateGraph(Pregel):
    """Compiled state graph - the runnable form of a StateGraph."""

    def __init__(self, *, builder: Any = None, **kwargs: Any):
        super().__init__(**{k: v for k, v in kwargs.items() if k != "builder"})
        self.builder = builder

    def get_state(self, config: RunnableConfig) -> StateSnapshot:
        return super().get_state(config)

    async def aget_state(self, config: RunnableConfig) -> StateSnapshot:
        return await super().aget_state(config)

    def update_state(
        self,
        config: RunnableConfig,
        values: dict[str, Any],
        as_node: Optional[str] = None,
    ) -> RunnableConfig:
        return super().update_state(config, values, as_node=as_node)

    def get_state_history(self, config: RunnableConfig) -> Any:
        raise NotImplementedError("get_state_history is not implemented")
