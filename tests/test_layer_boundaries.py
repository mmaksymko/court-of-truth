import ast
from pathlib import Path

ROOT = Path(__file__).parents[1] / "src" / "court"

CONFINED = {
    "torch": {"forensics"},
    "transformers": {"forensics"},
    "fastapi": {"api", "__main__.py"},
    "uvicorn": {"api", "__main__.py"},
    "starlette": {"api", "__main__.py"},
    "httpx": {"api", "ingest"},
    "bs4": {"ingest"},
    "openai": {"tribunal"},
    "agents": {"tribunal"},
}

ENV_ALLOWED = {"config.py"}


def _component(path: Path) -> str:
    rel = path.relative_to(ROOT).as_posix()
    return rel.split("/", 1)[0] if "/" in rel else rel


def _imports(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
    return names


def test_external_dependencies_stay_in_one_component() -> None:
    violations: set[str] = set()
    for path in ROOT.rglob("*.py"):
        component = _component(path)
        for module in _imports(ast.parse(path.read_text())):
            allowed = CONFINED.get(module.split(".", 1)[0])
            if allowed is not None and component not in allowed:
                violations.add(f"{component} imports {module.split('.', 1)[0]}")
    assert not violations, sorted(violations)


def test_lower_layers_do_not_import_the_api() -> None:
    violations: set[str] = set()
    for path in ROOT.rglob("*.py"):
        component = _component(path)
        if component in {"api", "__main__.py"}:
            continue
        for module in _imports(ast.parse(path.read_text())):
            if module.startswith("court.api"):
                violations.add(f"{component} imports {module}")
    assert not violations, sorted(violations)


def test_raw_environment_access_only_in_settings() -> None:
    violations: list[str] = []
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT).as_posix()
        if rel in ENV_ALLOWED:
            continue
        text = path.read_text()
        if "os.environ" in text or "os.getenv" in text:
            violations.append(rel)
    assert not violations, violations
