from pathlib import Path

ROOT = Path("geosc")
MAX_DEPTH = 4

lines = ["graph TD"]

def walk(path, parent=None, depth=0):
    if depth > MAX_DEPTH:
        return
    node_name = path.name.replace("-", "_")
    if parent:
        lines.append(f'    {parent} --> {node_name}')
    for p in path.iterdir():
        if p.is_dir() and not p.name.startswith("__"):
            walk(p, node_name, depth + 1)

walk(ROOT)

with open("docs/architecture.md", "w") as f:
    f.write("```mermaid\n")
    f.write("\n".join(lines))
    f.write("\n```")
