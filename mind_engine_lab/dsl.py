from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple
import ast
import math


class DSLError(Exception):
    pass


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


SAFE_FUNCS = {
    "min": min,
    "max": max,
    "abs": abs,
    "sqrt": math.sqrt,
    "log1p": math.log1p,
    "clamp": clamp,
}

SAFE_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Constant, ast.Name, ast.Load,
    ast.Call, ast.Compare, ast.BoolOp, ast.And, ast.Or, ast.IfExp,
    ast.Subscript, ast.Attribute, ast.Tuple, ast.List, ast.Dict,
)
SAFE_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
SAFE_UNARYOPS = (ast.UAdd, ast.USub, ast.Not)
SAFE_CMPOPS = (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)


@dataclass
class DSLTrace:
    expr: str
    bindings: Dict[str, Any]
    value: Any


class Namespace:
    """Attribute-access wrapper for dicts: ns.key -> dict['key'] (missing -> 0.0)"""
    def __init__(self, d: Dict[str, Any]):
        self._d = dict(d or {})

    def __getattr__(self, item: str):
        if item.startswith("_"):
            raise AttributeError(item)
        return self._d.get(item, 0.0)


def eval_expr(expr: str, bindings: Dict[str, Any]) -> Tuple[float, DSLTrace]:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise DSLError(f"DSL syntax error: {e}") from e

    _validate_ast(tree)
    val = _eval(tree.body, bindings)
    try:
        fval = float(val)
    except Exception as e:
        raise DSLError(f"DSL result not a number: {val!r}") from e

    return fval, DSLTrace(expr=expr, bindings=_shallow_sanitize(bindings), value=fval)


def _validate_ast(node: ast.AST):
    for n in ast.walk(node):
        if not isinstance(n, SAFE_NODES):
            raise DSLError(f"Disallowed expression node: {type(n).__name__}")
        if isinstance(n, ast.BinOp) and not isinstance(n.op, SAFE_BINOPS):
            raise DSLError(f"Disallowed binary operator: {type(n.op).__name__}")
        if isinstance(n, ast.UnaryOp) and not isinstance(n.op, SAFE_UNARYOPS):
            raise DSLError(f"Disallowed unary operator: {type(n.op).__name__}")
        if isinstance(n, ast.Compare):
            for op in n.ops:
                if not isinstance(op, SAFE_CMPOPS):
                    raise DSLError(f"Disallowed compare operator: {type(op).__name__}")
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Name):
                if n.func.id not in SAFE_FUNCS:
                    raise DSLError(f"Disallowed function: {n.func.id}")
            else:
                raise DSLError("Only direct function calls are allowed.")


def _eval(node: ast.AST, env: Dict[str, Any]):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Num):
        return node.n
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        if node.id in SAFE_FUNCS:
            return SAFE_FUNCS[node.id]
        raise DSLError(f"Unknown name: {node.id}")

    if isinstance(node, ast.BinOp):
        return _binop(node.op, _eval(node.left, env), _eval(node.right, env))

    if isinstance(node, ast.UnaryOp):
        return _unaryop(node.op, _eval(node.operand, env))

    if isinstance(node, ast.Call):
        fn = _eval(node.func, env)
        args = [_eval(a, env) for a in node.args]
        kwargs = {kw.arg: _eval(kw.value, env) for kw in node.keywords}
        return fn(*args, **kwargs)

    if isinstance(node, ast.Compare):
        left = _eval(node.left, env)
        ok = True
        for op, comp in zip(node.ops, node.comparators):
            right = _eval(comp, env)
            ok = ok and _cmp(op, left, right)
            left = right
        return ok

    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            return all(bool(_eval(v, env)) for v in node.values)
        if isinstance(node.op, ast.Or):
            return any(bool(_eval(v, env)) for v in node.values)

    if isinstance(node, ast.IfExp):
        return _eval(node.body, env) if bool(_eval(node.test, env)) else _eval(node.orelse, env)

    if isinstance(node, ast.Attribute):
        base = _eval(node.value, env)
        return getattr(base, node.attr)

    if isinstance(node, ast.Subscript):
        base = _eval(node.value, env)
        key = _eval(node.slice, env)
        return base[key]

    if isinstance(node, (ast.Tuple, ast.List)):
        return [_eval(elt, env) for elt in node.elts]

    if isinstance(node, ast.Dict):
        return {_eval(k, env): _eval(v, env) for k, v in zip(node.keys, node.values)}

    raise DSLError(f"Unsupported construct: {type(node).__name__}")


def _binop(op, a, b):
    if isinstance(op, ast.Add): return a + b
    if isinstance(op, ast.Sub): return a - b
    if isinstance(op, ast.Mult): return a * b
    if isinstance(op, ast.Div): return a / b
    if isinstance(op, ast.FloorDiv): return a // b
    if isinstance(op, ast.Mod): return a % b
    if isinstance(op, ast.Pow): return a ** b
    raise DSLError("Disallowed binary op")


def _unaryop(op, a):
    if isinstance(op, ast.UAdd): return +a
    if isinstance(op, ast.USub): return -a
    if isinstance(op, ast.Not): return not a
    raise DSLError("Disallowed unary op")


def _cmp(op, a, b):
    if isinstance(op, ast.Lt): return a < b
    if isinstance(op, ast.LtE): return a <= b
    if isinstance(op, ast.Gt): return a > b
    if isinstance(op, ast.GtE): return a >= b
    if isinstance(op, ast.Eq): return a == b
    if isinstance(op, ast.NotEq): return a != b
    raise DSLError("Disallowed compare op")


def _shallow_sanitize(env: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in env.items():
        if k in {"psych", "cog"} and isinstance(v, Namespace):
            out[k] = dict(v._d)
        elif isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
    return out