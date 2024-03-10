def get_max_h_value(dic):
  h_keys = [
    key for key in dic.keys()
    if key.startswith('Header ') and key.split(' ')[-1].isdigit() 
  ]
  if not h_keys:
      return None  # No keys with 'Header ' pattern found
  max_key = max(h_keys, key=lambda k: int(k.split(' ')[-1]))
  return dic[max_key]