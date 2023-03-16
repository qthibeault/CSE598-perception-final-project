from typing import Callable, List, Optional, Type, TypeVar, Union

from .callback_groups import CallbackGroup
from .context import Context
from .logging import RcutilsLogger
from .publisher import Publisher
from .qos import QosProfile
from .subscription import Subscription

MsgType = TypeVar("MsgType")

class Node:
    def __init__(
        self,
        node_name: str,
        *,
        context: Optional[Context] = ...,
        cli_args: List[str] = ...,
        namespace: Optional[str] = ...,
        use_global_arguments: bool = ...,
        enable_rosout: bool = ...,
        start_parameter_service: bool = ...,
        allow_undeclared_parameters: bool = ...,
        automatically_declare_parameters_from_overrides: bool = ...,
    ): ...
    def create_publisher(
        self,
        msg_type: MsgType,
        topic: str,
        qos_profile: Union[QosProfile, int],
        *,
        callback_group: Optional[CallbackGroup] = ...,
        event_callbacks: Optional[object] = ...,
    ) -> Publisher[MsgType]: ...
    def create_subscription(
        self,
        msg_type: Type[MsgType],
        topic: str,
        callback: Callable[[MsgType], None],
        qos_profile: Union[QosProfile, int],
        *,
        callback_group: Optional[CallbackGroup] = ...,
        event_callbacks: Optional[object] = ...,
        raw: bool = ...,
    ) -> Subscription: ...
    def get_logger(self) -> RcutilsLogger: ...
