import inspect
from IPython.display import Code, display
 
def display_source(func):
    """Dynamically displays the source code of a function."""
    source_code = inspect.getsource(func)
    display(Code(source_code, language='python'))
