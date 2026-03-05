# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import ast
import glob


def parse_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    classes = []
    functions = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods = []
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    doc = ast.get_docstring(item)
                    methods.append(
                        {
                            "name": item.name,
                            "args": [a.arg for a in item.args.args if a.arg != "self"],
                            "doc": doc,
                        }
                    )
            doc = ast.get_docstring(node)
            classes.append({"name": node.name, "doc": doc, "methods": methods})
        elif isinstance(node, ast.FunctionDef):
            doc = ast.get_docstring(node)
            functions.append(
                {"name": node.name, "args": [a.arg for a in node.args.args], "doc": doc}
            )

    return classes, functions


def generate_markdown(src_dir, output_file):
    md = "# SC-NeuroCore API Reference\n\n"

    files = glob.glob(os.path.join(src_dir, "**/*.py"), recursive=True)

    for filepath in sorted(files):
        if "__init__" in filepath:
            continue

        rel_path = os.path.relpath(filepath, src_dir)
        module_name = rel_path.replace(os.sep, ".").replace(".py", "")

        classes, functions = parse_file(filepath)

        if not classes and not functions:
            continue

        md += f"## Module `{module_name}`\n\n"

        for cls in classes:
            md += f"### Class `{cls['name']}`\n"
            if cls["doc"]:
                md += f"{cls['doc']}\n\n"

            for method in cls["methods"]:
                args = ", ".join(method["args"])
                md += f"- **{method['name']}**({args})\n"
                if method["doc"]:
                    md += f"  - {method['doc'].splitlines()[0]}\n"
            md += "\n"

        for func in functions:
            args = ", ".join(func["args"])
            md += f"### Function `{func['name']}({args})`\n"
            if func["doc"]:
                md += f"{func['doc']}\n\n"

        md += "---\n\n"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Generated {output_file}")


if __name__ == "__main__":
    src = os.path.join(os.path.dirname(__file__), "../src/sc_neurocore")
    out = os.path.join(os.path.dirname(__file__), "../API_REFERENCE.md")
    generate_markdown(src, out)
