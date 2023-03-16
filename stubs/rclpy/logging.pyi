class RcutilsLogger:
    def info(
        self,
        message: str,
        *,
        throttle_duration_sec: float = ...,
        throttle_time_source_type: str = ...,
        skip_first: bool = ...,
        once: bool = ...,
    ) -> None: ...
