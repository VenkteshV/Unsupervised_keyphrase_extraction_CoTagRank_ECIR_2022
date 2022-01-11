import os
import multiprocessing
import wsgi


_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..'))
_VAR = os.path.join(_ROOT, 'var')
_ETC = os.path.join(_ROOT, 'etc')

loglevel = 'info'
preload = True
worker_class="gthread"

# uncomment if needed 
# errorlog = os.path.join(_VAR, 'log/api-error.log')
# accesslog = os.path.join(_VAR, 'log/api-access.log')
errorlog = "-"
accesslog = "-"

# bind = 'unix:%s' % os.path.join(_VAR, 'run/gunicorn.sock')
bind = '0.0.0.0:5000'
# workers = 3
workers = multiprocessing.cpu_count() * 2 + 1
threads=workers
# multiprocessing.cpu_count() * 2 + 1

timeout = 30 * 60  # 6 minutes
keepalive = 24 * 60 * 60  # 1 day

capture_output = True