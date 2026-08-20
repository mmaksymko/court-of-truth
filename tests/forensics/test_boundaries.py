import ast
from pathlib import Path


def test_forensics_layer_has_no_llm_or_application_config_dependency() -> None:
    root = Path(__file__).parents[2] / "src" / "court" / "forensics"
    forbidden = ("agents", "openai", "court.config", "court.tribunal")
    violations: list[str] = []

    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            imported: list[str]
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported = [node.module or ""]
            else:
                continue
            for module in imported:
                if module in forbidden or module.startswith(
                    tuple(f"{prefix}." for prefix in forbidden)
                ):
                    violations.append(f"{path.relative_to(root)} imports {module}")

    assert not violations, violations


def test_runtime_schemas_do_not_depend_on_training() -> None:
    root = Path(__file__).parents[2] / "src" / "court" / "forensics"
    for path in root.rglob("*.py"):
        assert "training." not in path.read_text(), path
