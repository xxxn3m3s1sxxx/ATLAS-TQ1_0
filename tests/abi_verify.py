"""ABI verification: reconcile Python argtypes against C header.

Usage:
    from abi_verify import verify_abi
    verify_abi("atlas_ffi.h", dll)
    # Raises ValueError on mismatch — no crash, no 0x65.
"""

import ctypes
import re
import os

# C type → ctypes mapping for parameter types used in atlas_ffi.h
C_TO_CTYPES = {
    'void': None,
    'int': ctypes.c_int,
    'float': ctypes.c_float,
    'uint64_t': ctypes.c_uint64,
    'uint8_t': ctypes.c_uint8,
    'uint16_t': ctypes.c_uint16,
    'uint32_t': ctypes.c_uint32,
    'int8_t': ctypes.c_int8,
    'int32_t': ctypes.c_int32,
    'size_t': ctypes.c_size_t,
    'bool': ctypes.c_bool,
    'void*': ctypes.c_void_p,
    'char*': ctypes.c_char_p,
    'const char*': ctypes.c_char_p,
    'const uint8_t*': ctypes.POINTER(ctypes.c_uint8),
    'const int*': ctypes.POINTER(ctypes.c_int),
    'const float*': ctypes.POINTER(ctypes.c_float),
    'const int32_t*': ctypes.POINTER(ctypes.c_int32),
    'const int32_t**': ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)),
    'int*': ctypes.POINTER(ctypes.c_int),
    'float*': ctypes.POINTER(ctypes.c_float),
    'uint8_t*': ctypes.POINTER(ctypes.c_uint8),
    'const uint16_t*': ctypes.POINTER(ctypes.c_uint16),
}

# Function pointer callbacks — skip typedef, but track expected params
CALLBACK_TYPES = {
    'atlas_token_callback': [ctypes.c_int, ctypes.c_void_p],
}

# Struct return types — opaque, treated as void
STRUCT_RETURNS = {'AtlasModelConfig'}


def _normalize_type(s: str) -> str:
    """Collapse pointer spacing, strip names for map lookup."""
    s = s.strip()
    s = re.sub(r'\s*\*\s*', '*', s)
    s = re.sub(r'\s+', ' ', s)
    return s


def _param_type_to_ctypes(param_str: str):
    """Parse a C param declaration like 'const int* foo' → ctypes type."""
    s = param_str.strip()
    if not s or s == 'void':
        return None

    # Strip parameter name: try last-token-stripped first
    # e.g. "const int* input_ids" → try "const int*", fallback to full string
    tokens = s.split()
    if len(tokens) > 1:
        last = tokens[-1].replace('*', '')
        if last.isidentifier() and last not in ('const', 'unsigned', 'signed', 'long', 'short'):
            type_guess = _normalize_type(' '.join(tokens[:-1]))
            if type_guess in C_TO_CTYPES:
                return C_TO_CTYPES[type_guess]
            no_const = re.sub(r'\bconst\s+', '', type_guess).strip()
            if no_const in C_TO_CTYPES:
                return C_TO_CTYPES[no_const]

    normalized = _normalize_type(s)

    # Try direct match (with const)
    if normalized in C_TO_CTYPES:
        return C_TO_CTYPES[normalized]

    # Try without const
    no_const = re.sub(r'\bconst\s+', '', normalized).strip()
    if no_const in C_TO_CTYPES:
        return C_TO_CTYPES[no_const]

    # Try just the first word (base type)
    base = normalized.split()[0] if normalized.split() else normalized
    if base in C_TO_CTYPES:
        return C_TO_CTYPES[base]

    return None


def parse_header(header_path: str):
    """Extract {func_name: [(type_str, ctypes_type), ...]} from atlas_ffi.h.

    Handles multi-line declarations, strips comments.
    Skips typedefs and non-ATLAS_API declarations.
    """
    if not os.path.exists(header_path):
        raise FileNotFoundError(f"Header not found: {header_path}")

    text = open(header_path, 'r', encoding='utf-8').read()
    text = re.sub(r'//.*', '', text)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)

    # Join continuation lines: collect text between ATLAS_API and );
    lines = text.split('\n')
    decls = []
    buf = None
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s.startswith('ATLAS_API '):
            buf = s
        elif buf is not None:
            buf += ' ' + s
        if buf is not None and buf.rstrip().endswith(';'):
            decls.append(buf)
            buf = None

    funcs = {}
    for decl in decls:
        # Skip typedefs
        if 'typedef' in decl:
            continue

        # Extract: ATLAS_API <ret> <name>(<params>);
        m = re.match(
            r'ATLAS_API\s+'
            r'(?:const\s+)?(?:struct\s+)?(\w+(?:\s*\*)?)\s+'  # return type
            r'(\w+)\s*'  # function name
            r'\(([^)]*)\)\s*;',  # params
            decl,
        )
        if not m:
            continue

        ret_type, name, params_str = m.groups()
        name = name.strip()

        # Skip callbacks and struct returns (not direct API calls)
        if name in CALLBACK_TYPES:
            funcs[name] = CALLBACK_TYPES[name]
            continue

        # Parse each parameter
        params = []
        for p in params_str.split(','):
            p = p.strip()
            if not p:
                continue
            ctype = _param_type_to_ctypes(p)
            if ctype is not None or p == 'void':
                params.append(ctype)

        funcs[name] = params

    return funcs


def verify_abi(header_path: str, dll: ctypes.CDLL, func_names: list = None):
    """Compare Python argtypes against C header for listed functions.

    Raises ValueError with details on first mismatch.
    If func_names is None, checks all Python-bound functions.
    """
    expected = parse_header(header_path)

    # Discover Python-bound functions (skip private/dunder)
    py_funcs = func_names or [
        name for name in dir(dll)
        if not name.startswith('_')
        and hasattr(getattr(dll, name), 'argtypes')
    ]

    errors = []
    for name in sorted(py_funcs):
        py_func = getattr(dll, name, None)
        if py_func is None:
            errors.append(f"{name}: not found on DLL")
            continue
        if not hasattr(py_func, 'argtypes') or py_func.argtypes is None:
            errors.append(f"{name}: no argtypes set on Python side")
            continue

        if name not in expected:
            errors.append(f"{name}: not found in C header")
            continue

        py_types = list(py_func.argtypes)
        c_types = expected[name]

        if len(py_types) != len(c_types):
            errors.append(
                f"{name}: parameter count mismatch — "
                f"Python {len(py_types)} {_types_str(py_types)} vs "
                f"C header {len(c_types)} {_types_str(c_types)}"
            )
            continue

        # Check each parameter type
        for i, (py_t, c_t) in enumerate(zip(py_types, c_types)):
            if c_t is None:
                continue  # void/unknown — skip check
            if py_t is not c_t:
                # Special case: POINTER types may compare by base type
                if _is_pointer(py_t) and _is_pointer(c_t):
                    if not _same_pointer_base(py_t, c_t):
                        errors.append(
                            f"{name}[param {i}]: pointer type mismatch — "
                            f"Python {py_t} vs C expected {c_t}"
                        )
                else:
                    errors.append(
                        f"{name}[param {i}]: type mismatch — "
                        f"Python {py_t} vs C expected {c_t}"
                    )

    if errors:
        raise ValueError(
            "ABI MISMATCH — aborting before first C call:\n  "
            + "\n  ".join(errors)
        )

    return True


def _types_str(types: list) -> str:
    return "[" + ", ".join(
        t.__name__ if hasattr(t, '__name__') else str(t) for t in types
    ) + "]"


_POINTER_TYPE = type(ctypes.POINTER(ctypes.c_int))


def _is_pointer(t) -> bool:
    return isinstance(t, _POINTER_TYPE) or type(t).__name__ == 'PyCPointerType'


def _same_pointer_base(a, b) -> bool:
    """Check if two POINTER types point to the same base type."""
    try:
        return a._type_ is b._type_
    except AttributeError:
        return False
