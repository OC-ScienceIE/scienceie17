from os.path import join, abspath, dirname

REPOS_DIR = abspath(join(dirname(__file__), '..', '..', '..'))
DATA_DIR = join(REPOS_DIR, 'data')
LOCAL_DIR = join(REPOS_DIR, '_local')

ENTITIES = 'Material', 'Process', 'Task'


