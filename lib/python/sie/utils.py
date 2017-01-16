from glob import glob

def sorted_glob(pattern):
    """
    glob which return sorted results, because otherwise order depends on OS
    """
    return sorted(glob(pattern))