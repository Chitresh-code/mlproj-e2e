import os
print(os.path.abspath('src'))
print(os.path.abspath(os.path.join('src', 'exception.py')))
print(os.path.abspath(os.path.join('src', 'logger.py')))