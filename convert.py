import os
import subprocess

def convert_files(directory, direction="py_to_ipynb"):
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if direction == "py_to_ipynb" and file.endswith(".py"):
                print(f"Converting {file_path} to .ipynb")
                subprocess.run(["jupytext", "--to", "notebook", file_path])
            elif direction == "ipynb_to_py" and file.endswith(".ipynb"):
                print(f"Converting {file_path} to .py")
                subprocess.run(["jupytext", "--to", "py", file_path])

if __name__ == "__main__":
    project_dir = "."
    convert_files(project_dir, direction="py_to_ipynb")