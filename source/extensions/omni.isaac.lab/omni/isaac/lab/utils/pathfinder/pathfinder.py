import os

def find_absolute_path(relative_path):
    home_dir = os.path.expanduser('~')
    for dirpath, dirnames, filenames in os.walk(home_dir):
        potential_path = os.path.join(dirpath, relative_path)
        if os.path.exists(potential_path):
            return os.path.abspath(potential_path)
    raise RuntimeError("Unable to find absolute path of ", relative_path)