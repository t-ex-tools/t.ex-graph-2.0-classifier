import config

excluder = lambda x: x.lower() not in config.excluded_rows

def is_continuous(type):
  return type == 'float64'