


def truncate_float(float_number, decimal_places):
    multiplier = 10 ** decimal_places
    return int(float_number * multiplier) / multiplier
    
    
float_number=.985353467436523
decimal_places=5

truncated=truncate_float(float_number, decimal_places)
print(truncated)
