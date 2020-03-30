import datetime
from dateutil.parser import parse

def convert_time(input_times):

    # check for non-iterable inputs
    try:
        x = iter(input_times)
    except:
        input_times = [input_times]

    # parse input and convert to datetime
    out = []
    for t in input_times:
        if type(t) != 'str':
            t = str(t)
        d = parse(t)
        out.append(d)

    # return a scalar if a scalar was input
    if len(out) == 1:
        out = out[0]

    return out

