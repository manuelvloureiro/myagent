class GraphRecursionError(RecursionError):
    """Raised when the graph exceeds the recursion limit."""

    pass


class InvalidUpdateError(Exception):
    """Raised when a channel update is invalid."""

    pass


class EmptyChannelError(Exception):
    """Raised when reading from an empty channel."""

    pass


class NodeInterrupt(Exception):
    """Raised when a node is interrupted."""

    pass
