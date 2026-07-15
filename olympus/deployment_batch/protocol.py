"""Small length-prefixed JSON protocol used by batch deployment."""

import json
import socket
import struct

_HEADER = struct.Struct("!I")
MAX_MESSAGE = 4 * 1024 * 1024


def _read_exact(sock, size):
    chunks = []
    left = size
    while left:
        data = sock.recv(left)
        if not data:
            raise EOFError("inference connection closed")
        chunks.append(data)
        left -= len(data)
    return b"".join(chunks)


def send_message(sock: socket.socket, message: dict) -> None:
    body = json.dumps(message, separators=(",", ":"), allow_nan=False).encode()
    if len(body) > MAX_MESSAGE:
        raise ValueError("inference message is too large")
    sock.sendall(_HEADER.pack(len(body)) + body)


def recv_message(sock: socket.socket) -> dict:
    size = _HEADER.unpack(_read_exact(sock, _HEADER.size))[0]
    if size > MAX_MESSAGE:
        raise ValueError("inference message is too large")
    value = json.loads(_read_exact(sock, size))
    if not isinstance(value, dict):
        raise ValueError("inference message must be an object")
    return value
