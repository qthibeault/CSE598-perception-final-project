from typing import Generic, TypeVar, Union

MsgType = TypeVar("MsgType")

class Publisher(Generic[MsgType]):
    def publish(self, msg: Union[MsgType, bytes]) -> None: ...
