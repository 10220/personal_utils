"""
Useful Functions
"""


def update_progress(step, total_steps=None):
    """
    Adds a simple progress bar to keep track of a loop. Example:
        
        update_progress(0)
        for i in range(number_of_elements):
            example_function(i)
            update_progress(i, number_of_elements)
    """
    import os

    try:
        progress = (step+1)/total_steps
    except TypeError:
        progress = 0

    bar_length = 20
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1
    block = int(round(bar_length * progress))
    text = "Progress: [{0}] {1:.1f}%".format(
        "#" * block + "-" * (bar_length - block), progress * 100
    )
    if os.environ["_"].endswith("jupyter-notebook"):
        from IPython.display import clear_output
        clear_output(wait=True)
        print(text)
    else:
        import sys
        sys.stdout.write("\r" + text)
    return
