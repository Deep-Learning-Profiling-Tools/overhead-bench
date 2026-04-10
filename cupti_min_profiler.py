import ctypes
import ctypes.util
import os
import re
from dataclasses import dataclass, field
from typing import Dict, Optional


_BUFFER_SIZE = 1 << 20
_BUFFER_ALIGNMENT = 8

_CUPTI_SUCCESS = 0
_CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL = 10
_CUPTI_ACTIVITY_FLAG_FLUSH_FORCED = 1
_CUPTI_CB_DOMAIN_DRIVER_API = 1
_CUPTI_CB_DOMAIN_RUNTIME_API = 2
_CUPTI_API_ENTER = 0

_KERNEL_ACTIVITY_KINDS = {
    _CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,
}

_KERNEL_CALLBACK_SYMBOLS = {
    _CUPTI_CB_DOMAIN_RUNTIME_API: (
        "CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000",
        "CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_ptsz_v7000",
        "CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_v11060",
        "CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_ptsz_v11060",
        "CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020",
        "CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_ptsz_v7000",
    ),
    _CUPTI_CB_DOMAIN_DRIVER_API: (
        "CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel",
        "CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz",
        "CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx",
        "CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz",
        "CUPTI_DRIVER_TRACE_CBID_cuLaunch",
        "CUPTI_DRIVER_TRACE_CBID_cuLaunchGrid",
        "CUPTI_DRIVER_TRACE_CBID_cuLaunchGridAsync",
    ),
}
_GRAPH_CALLBACK_SYMBOLS = {
    _CUPTI_CB_DOMAIN_RUNTIME_API: (
        "CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_v10000",
        "CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_ptsz_v10000",
    ),
    _CUPTI_CB_DOMAIN_DRIVER_API: (
        "CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch",
    ),
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


class _CallbackData(ctypes.Structure):
    _fields_ = [
        ("callbackSite", ctypes.c_int),
        ("context", ctypes.c_void_p),
        ("contextUid", ctypes.c_uint32),
        ("correlationData", ctypes.POINTER(ctypes.c_uint64)),
        ("correlationId", ctypes.c_uint32),
        ("functionName", ctypes.c_char_p),
        ("functionParams", ctypes.c_void_p),
        ("functionReturnValue", ctypes.c_void_p),
        ("symbolName", ctypes.c_char_p),
    ]


@dataclass
class _Session:
    name: str
    lib: ctypes.CDLL
    buffers: Dict[int, ctypes.Array] = field(default_factory=dict)
    enabled_kinds: list[int] = field(default_factory=list)
    subscriber: Optional[ctypes.c_void_p] = None
    enabled_callbacks: list[tuple[int, int]] = field(default_factory=list)
    callback_kinds: Dict[tuple[int, int], str] = field(default_factory=dict)
    kernel_records: int = 0
    dropped_records: int = 0
    kernel_launch_callbacks: int = 0
    graph_launch_callbacks: int = 0
    callback_error: Optional[Exception] = None
    request_cb: Optional[object] = None
    complete_cb: Optional[object] = None
    api_callback: Optional[object] = None


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


def _candidate_include_paths() -> list[str]:
    paths = []
    override = os.environ.get("CUPTI_INCLUDE_PATH")
    if override:
        paths.append(override)

    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        paths.append(os.path.join(cuda_home, "extras", "CUPTI", "include"))
        paths.append(os.path.join(cuda_home, "include"))

    paths.extend(
        [
            "/usr/local/cuda/extras/CUPTI/include",
            "/usr/local/cuda/include",
            "/usr/include",
        ]
    )

    unique_paths = []
    seen = set()
    for path in paths:
        if path not in seen:
            unique_paths.append(path)
            seen.add(path)
    return unique_paths


def _read_header(filename: str) -> Optional[str]:
    for include_dir in _candidate_include_paths():
        header_path = os.path.join(include_dir, filename)
        try:
            with open(header_path, "r", encoding="utf-8", errors="replace") as handle:
                return handle.read()
        except OSError:
            continue
    return None


def _parse_cbid_values(header_text: str, symbols: tuple[str, ...]) -> tuple[int, ...]:
    values = []
    for symbol in symbols:
        match = re.search(rf"\b{re.escape(symbol)}\s*=\s*(\d+)", header_text)
        if match is not None:
            values.append(int(match.group(1)))
    return tuple(values)


def _resolve_callback_ids() -> tuple[Dict[int, tuple[int, ...]], Dict[int, tuple[int, ...]]]:
    kernel_ids: Dict[int, tuple[int, ...]] = {}
    graph_ids: Dict[int, tuple[int, ...]] = {}

    runtime_header = _read_header("cupti_runtime_cbid.h")
    if runtime_header is not None:
        ids = _parse_cbid_values(runtime_header, _KERNEL_CALLBACK_SYMBOLS[_CUPTI_CB_DOMAIN_RUNTIME_API])
        if ids:
            kernel_ids[_CUPTI_CB_DOMAIN_RUNTIME_API] = ids
        ids = _parse_cbid_values(runtime_header, _GRAPH_CALLBACK_SYMBOLS[_CUPTI_CB_DOMAIN_RUNTIME_API])
        if ids:
            graph_ids[_CUPTI_CB_DOMAIN_RUNTIME_API] = ids

    driver_header = _read_header("cupti_driver_cbid.h")
    if driver_header is not None:
        ids = _parse_cbid_values(driver_header, _KERNEL_CALLBACK_SYMBOLS[_CUPTI_CB_DOMAIN_DRIVER_API])
        if ids:
            kernel_ids[_CUPTI_CB_DOMAIN_DRIVER_API] = ids
        ids = _parse_cbid_values(driver_header, _GRAPH_CALLBACK_SYMBOLS[_CUPTI_CB_DOMAIN_DRIVER_API])
        if ids:
            graph_ids[_CUPTI_CB_DOMAIN_DRIVER_API] = ids

    return kernel_ids, graph_ids


def _configure_api(lib: ctypes.CDLL) -> None:
    lib.cuptiSubscribe.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.cuptiSubscribe.restype = ctypes.c_int

    lib.cuptiUnsubscribe.argtypes = [ctypes.c_void_p]
    lib.cuptiUnsubscribe.restype = ctypes.c_int

    lib.cuptiEnableCallback.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_uint32,
    ]
    lib.cuptiEnableCallback.restype = ctypes.c_int

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


def _build_api_callback(session: _Session) -> object:
    callback_t = ctypes.CFUNCTYPE(
        None,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_uint32,
        ctypes.c_void_p,
    )

    def callback(userdata, domain, cbid, cbdata):
        del userdata
        try:
            if cbdata is None:
                return
            callback_data = ctypes.cast(cbdata, ctypes.POINTER(_CallbackData)).contents
            if callback_data.callbackSite != _CUPTI_API_ENTER:
                return

            callback_kind = session.callback_kinds.get((domain, cbid))
            if callback_kind == "kernel":
                session.kernel_launch_callbacks += 1
            elif callback_kind == "graph":
                session.graph_launch_callbacks += 1
        except Exception as exc:
            if session.callback_error is None:
                session.callback_error = exc

    return callback_t(callback)


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


def _subscribe_launch_callbacks(session: _Session) -> None:
    kernel_ids, graph_ids = _resolve_callback_ids()
    if not kernel_ids and not graph_ids:
        raise RuntimeError(
            "Failed to resolve CUPTI callback IDs for kernel or graph launches. "
            "Set CUPTI_INCLUDE_PATH to the directory containing cupti_runtime_cbid.h "
            "and cupti_driver_cbid.h."
        )

    subscriber = ctypes.c_void_p()
    api_callback = _build_api_callback(session)
    _check_result(
        session.lib,
        session.lib.cuptiSubscribe(ctypes.byref(subscriber), api_callback, None),
        "cuptiSubscribe",
    )
    session.subscriber = subscriber
    session.api_callback = api_callback

    try:
        for callback_kind, callback_ids in (("kernel", kernel_ids), ("graph", graph_ids)):
            for domain, ids in callback_ids.items():
                for cbid in ids:
                    _check_result(
                        session.lib,
                        session.lib.cuptiEnableCallback(1, subscriber, domain, cbid),
                        f"cuptiEnableCallback(domain={domain}, cbid={cbid})",
                    )
                    session.enabled_callbacks.append((domain, cbid))
                    session.callback_kinds[(domain, cbid)] = callback_kind
    except Exception:
        for domain, cbid in reversed(session.enabled_callbacks):
            session.lib.cuptiEnableCallback(0, subscriber, domain, cbid)
        session.enabled_callbacks.clear()
        session.callback_kinds.clear()
        session.lib.cuptiUnsubscribe(subscriber)
        session.subscriber = None
        session.api_callback = None
        raise


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
        _subscribe_launch_callbacks(session)
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
        if session.subscriber is not None:
            for domain, cbid in reversed(session.enabled_callbacks):
                lib.cuptiEnableCallback(0, session.subscriber, domain, cbid)
            lib.cuptiUnsubscribe(session.subscriber)
            session.enabled_callbacks.clear()
            session.callback_kinds.clear()
            session.subscriber = None
            session.api_callback = None
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

    if session.subscriber is not None:
        for domain, cbid in reversed(session.enabled_callbacks):
            try:
                _check_result(
                    session.lib,
                    session.lib.cuptiEnableCallback(0, session.subscriber, domain, cbid),
                    f"cuptiEnableCallback(disable, domain={domain}, cbid={cbid})",
                )
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        try:
            _check_result(
                session.lib,
                session.lib.cuptiUnsubscribe(session.subscriber),
                "cuptiUnsubscribe",
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
        f"dropped_records={session.dropped_records} "
        f"kernel_launch_callbacks={session.kernel_launch_callbacks} "
        f"graph_launch_callbacks={session.graph_launch_callbacks}",
        flush=True,
    )
