import ctypes
import ctypes.util
import os
from dataclasses import dataclass, field
from typing import Dict, Optional


_BUFFER_SIZE = 1 << 20
_BUFFER_ALIGNMENT = 8

_CUPTI_SUCCESS = 0
_CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL = 10
_CUPTI_ACTIVITY_FLAG_FLUSH_FORCED = 1

_KERNEL_ACTIVITY_KINDS = {
    _CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,
}

_SESSION: Optional["_Session"] = None

_REQUEST_CB_T = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_size_t),
)
_COMPLETE_CB_T = ctypes.CFUNCTYPE(
    None,
    ctypes.c_void_p,
    ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_size_t,
    ctypes.c_size_t,
)


class _ActivityRecord(ctypes.Structure):
    _fields_ = [("kind", ctypes.c_int)]


@dataclass
class _Session:
    name: str
    lib: ctypes.CDLL
    buffers: Dict[int, ctypes.Array] = field(default_factory=dict)
    enabled_kinds: list[int] = field(default_factory=list)
    kernel_records: int = 0
    dropped_records: int = 0
    callback_error: Optional[Exception] = None
    request_cb: Optional[object] = None
    complete_cb: Optional[object] = None


def _candidate_library_paths() -> list[str]:
    paths = []
    override = os.environ.get("CUPTI_LIB_PATH")
    if override:
        paths.append(override)

    found = ctypes.util.find_library("cupti")
    if found:
        paths.append(found)

    paths.extend(
        [
            "/usr/local/cuda/extras/CUPTI/lib64/libcupti.so",
            "/usr/local/cuda/lib64/libcupti.so",
            "/usr/lib/x86_64-linux-gnu/libcupti.so",
            "/usr/lib/wsl/lib/libcupti.so",
        ]
    )

    unique_paths = []
    seen = set()
    for path in paths:
        if path not in seen:
            unique_paths.append(path)
            seen.add(path)
    return unique_paths


def _load_cupti_library() -> ctypes.CDLL:
    errors = []
    for path in _candidate_library_paths():
        try:
            return ctypes.CDLL(path)
        except OSError as exc:
            errors.append(f"{path}: {exc}")

    details = "; ".join(errors) if errors else "no candidate library paths found"
    raise RuntimeError(
        "CUPTI library is unavailable. Set CUPTI_LIB_PATH to libcupti.so. "
        f"Load attempts: {details}"
    )


def _configure_api(lib: ctypes.CDLL) -> None:
    lib.cuptiGetResultString.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_char_p),
    ]
    lib.cuptiGetResultString.restype = ctypes.c_int

    lib.cuptiActivityRegisterCallbacks.argtypes = [_REQUEST_CB_T, _COMPLETE_CB_T]
    lib.cuptiActivityRegisterCallbacks.restype = ctypes.c_int

    lib.cuptiActivityEnable.argtypes = [ctypes.c_int]
    lib.cuptiActivityEnable.restype = ctypes.c_int

    lib.cuptiActivityDisable.argtypes = [ctypes.c_int]
    lib.cuptiActivityDisable.restype = ctypes.c_int

    lib.cuptiActivityGetNextRecord.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.cuptiActivityGetNextRecord.restype = ctypes.c_int

    lib.cuptiActivityGetNumDroppedRecords.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.cuptiActivityGetNumDroppedRecords.restype = ctypes.c_int

    lib.cuptiActivityFlushAll.argtypes = [ctypes.c_uint32]
    lib.cuptiActivityFlushAll.restype = ctypes.c_int


def _result_string(lib: ctypes.CDLL, result: int) -> str:
    result_str = ctypes.c_char_p()
    status = lib.cuptiGetResultString(result, ctypes.byref(result_str))
    if status == _CUPTI_SUCCESS and result_str.value:
        return result_str.value.decode("utf-8", errors="replace")
    return f"CUPTI result {result}"


def _is_max_limit_reached(lib: ctypes.CDLL, result: int) -> bool:
    result_text = _result_string(lib, result).lower().replace("_", " ")
    return "max limit reached" in result_text


def _check_result(lib: ctypes.CDLL, result: int, operation: str) -> None:
    if result != _CUPTI_SUCCESS:
        raise RuntimeError(f"{operation} failed: {_result_string(lib, result)}")


def _aligned_buffer() -> tuple[ctypes.Array, ctypes.POINTER(ctypes.c_uint8)]:
    raw = (ctypes.c_uint8 * (_BUFFER_SIZE + _BUFFER_ALIGNMENT))()
    address = ctypes.addressof(raw)
    aligned = (address + (_BUFFER_ALIGNMENT - 1)) & ~(_BUFFER_ALIGNMENT - 1)
    ptr = ctypes.cast(ctypes.c_void_p(aligned), ctypes.POINTER(ctypes.c_uint8))
    return raw, ptr


def _consume_buffer(session: _Session, buffer: ctypes.POINTER(ctypes.c_uint8), valid_size: int) -> None:
    record_ptr = ctypes.c_void_p()
    while True:
        result = session.lib.cuptiActivityGetNextRecord(
            buffer,
            valid_size,
            ctypes.byref(record_ptr),
        )
        if result == _CUPTI_SUCCESS:
            record = ctypes.cast(record_ptr, ctypes.POINTER(_ActivityRecord)).contents
            if record.kind in _KERNEL_ACTIVITY_KINDS:
                session.kernel_records += 1
            continue
        if _is_max_limit_reached(session.lib, result):
            return
        raise RuntimeError(
            "cuptiActivityGetNextRecord failed: "
            f"{_result_string(session.lib, result)}"
        )


def _build_callbacks(session: _Session) -> tuple[object, object]:
    def request_buffer(buffer_ptr, size_ptr, max_records_ptr):
        raw, aligned_ptr = _aligned_buffer()
        buffer_key = ctypes.cast(aligned_ptr, ctypes.c_void_p).value
        session.buffers[buffer_key] = raw
        buffer_ptr[0] = aligned_ptr
        size_ptr[0] = _BUFFER_SIZE
        max_records_ptr[0] = 0
        return _CUPTI_SUCCESS

    def complete_buffer(ctx, stream_id, buffer_ptr, size, valid_size):
        buffer_key = 0
        if bool(buffer_ptr):
            buffer_key = ctypes.cast(buffer_ptr, ctypes.c_void_p).value or 0
        try:
            if bool(buffer_ptr) and valid_size:
                _consume_buffer(session, buffer_ptr, valid_size)

            dropped = ctypes.c_size_t()
            _check_result(
                session.lib,
                session.lib.cuptiActivityGetNumDroppedRecords(
                    ctx,
                    stream_id,
                    ctypes.byref(dropped),
                ),
                "cuptiActivityGetNumDroppedRecords",
            )
            session.dropped_records += dropped.value
        except Exception as exc:
            if session.callback_error is None:
                session.callback_error = exc
        finally:
            if buffer_key:
                session.buffers.pop(buffer_key, None)

    return _REQUEST_CB_T(request_buffer), _COMPLETE_CB_T(complete_buffer)


def start(name: str, mode=None, hook=None) -> _Session:
    del mode, hook

    global _SESSION
    if _SESSION is not None:
        return _SESSION

    lib = _load_cupti_library()
    _configure_api(lib)
    session = _Session(name=name, lib=lib)
    request_cb, complete_cb = _build_callbacks(session)
    session.request_cb = request_cb
    session.complete_cb = complete_cb

    try:
        _check_result(
            lib,
            lib.cuptiActivityRegisterCallbacks(request_cb, complete_cb),
            "cuptiActivityRegisterCallbacks",
        )
        for kind in (
            _CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,
        ):
            _check_result(lib, lib.cuptiActivityEnable(kind), f"cuptiActivityEnable({kind})")
            session.enabled_kinds.append(kind)
    except Exception:
        for kind in reversed(session.enabled_kinds):
            lib.cuptiActivityDisable(kind)
        raise

    _SESSION = session
    return session


def finalize() -> None:
    global _SESSION
    session = _SESSION
    if session is None:
        return

    _SESSION = None

    first_error: Optional[Exception] = session.callback_error
    try:
        if first_error is None:
            _check_result(
                session.lib,
                session.lib.cuptiActivityFlushAll(_CUPTI_ACTIVITY_FLAG_FLUSH_FORCED),
                "cuptiActivityFlushAll",
            )
    except Exception as exc:
        first_error = exc

    for kind in reversed(session.enabled_kinds):
        try:
            _check_result(
                session.lib,
                session.lib.cuptiActivityDisable(kind),
                f"cuptiActivityDisable({kind})",
            )
        except Exception as exc:
            if first_error is None:
                first_error = exc

    if session.callback_error is not None and first_error is None:
        first_error = session.callback_error

    if first_error is not None:
        raise first_error

    print(
        f"[cupti_min] kernel_records={session.kernel_records} "
        f"dropped_records={session.dropped_records}",
        flush=True,
    )
