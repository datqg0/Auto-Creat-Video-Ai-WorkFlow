"""Static policy cho code Manim do LLM sinh ra.

Đây là lớp defense-in-depth. Cô lập thật sự vẫn do container runtime đảm nhiệm.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field


_ALLOWED_IMPORT_ROOTS = {"manim", "numpy", "math", "random"}
_BLOCKED_NAMES = {
    "open", "eval", "exec", "compile", "__import__", "input", "breakpoint",
    "globals", "locals", "vars", "getattr", "setattr", "delattr",
}
_BLOCKED_ROOTS = {
    "os", "sys", "subprocess", "socket", "pathlib", "shutil", "requests",
    "urllib", "http", "ftplib", "importlib", "builtins", "ctypes", "pickle",
}


@dataclass(slots=True)
class ValidationResult:
    allowed: bool
    errors: list[str] = field(default_factory=list)


class _PolicyVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.has_scene = False
        self.has_construct = False
        self.node_count = 0

    def generic_visit(self, node: ast.AST) -> None:
        self.node_count += 1
        if self.node_count > 6000:
            self.errors.append("Code vượt giới hạn 6000 AST nodes.")
            return
        super().generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for alias in node.names:
            root = alias.name.split(".", 1)[0]
            if root not in _ALLOWED_IMPORT_ROOTS:
                self.errors.append(f"Import không được phép: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        root = (node.module or "").split(".", 1)[0]
        if root not in _ALLOWED_IMPORT_ROOTS:
            self.errors.append(f"Import không được phép: {node.module or '<relative>'}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        if isinstance(node.func, ast.Name) and node.func.id in _BLOCKED_NAMES:
            self.errors.append(f"Lời gọi bị chặn: {node.func.id}()")
        if isinstance(node.func, ast.Attribute):
            root = node.func.value
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id in _BLOCKED_ROOTS:
                self.errors.append(f"API bị chặn: {root.id}.{node.func.attr}()")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
        if node.id in _BLOCKED_ROOTS:
            self.errors.append(f"Tên module bị chặn: {node.id}")

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        if node.attr.startswith("__"):
            self.errors.append(f"Dunder attribute bị chặn: {node.attr}")
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:  # noqa: N802
        self.errors.append("Vòng while không được phép trong generated scene.")
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        bases = {
            base.id for base in node.bases if isinstance(base, ast.Name)
        }
        if "Scene" in bases or any(name.endswith("Scene") for name in bases):
            self.has_scene = True
            self.has_construct = any(
                isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "construct"
                for item in node.body
            )
        self.generic_visit(node)


def validate_manim_code(code: str) -> ValidationResult:
    """Chỉ cho phép code dựng Scene; không cho phép filesystem, network hay process API."""
    if not code or not code.strip():
        return ValidationResult(False, ["Code Manim rỗng."])
    if len(code) > 120_000:
        return ValidationResult(False, ["Code Manim vượt giới hạn 120000 ký tự."])
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return ValidationResult(False, [f"SyntaxError: {exc}"])

    visitor = _PolicyVisitor()
    visitor.visit(tree)
    if not visitor.has_scene:
        visitor.errors.append("Không tìm thấy class kế thừa Manim Scene.")
    elif not visitor.has_construct:
        visitor.errors.append("Scene không có construct(self).")
    # Giữ thứ tự nhưng loại lỗi trùng để log gọn.
    errors = list(dict.fromkeys(visitor.errors))
    return ValidationResult(not errors, errors)
