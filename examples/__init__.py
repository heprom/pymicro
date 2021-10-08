import importlib
import pathlib as pl 
from typing import List


PYMICRO_EXAMPLES_DATA_DIR = pl.Path(__file__).parent / "data" 

def list_examples() -> List[str]:
    """Generate the list of available examples

    Returns:
        List[str]: the list of available examples
    """
    example_dir = pl.Path(__file__).parent
    all_example = []
    for child in example_dir.glob("*"):
        if( child.is_dir() ):
            subdir_name = child.name
            for py in child.glob("*.py"):
                if py.name == "__init__.py":
                    continue
                all_example.append( f"{subdir_name}.{py.name.replace('.py', '')}")

    return all_example

def run_example( name: str) -> None:
    """Run an example.

    Args:
        name (str): Name of the example to run
    """

    if not name in list_examples():
        raise Exception(f"Example {name} doesn't exists") 

    try:
        importlib.import_module(f"pymicro.examples.{name}")
    except Exception as e:
        print(f"Failed to execute example {name}.\n {e.args}")

