
# one-step ahead forecasting functions from a previous RST session 

def fc_naive(data, **kwargs):
    """The 'naive' forecast of the next point in `data` (presumed to be 
    ordered in time) is equal to the last point observed in the series.
    
    `data` should be a 1-D numpy array
    
    Returns a single-valued forecast for the next value in the series.
    """
    forecast = data[-1]
    return forecast


def fc_snaive(data, n=7, **kwargs):
    """The 'seasonal naive' forecast of the next point in `data` (presumed to be 
    ordered in time) is equal to the point observed `n` points prior in the series.
    
    `data` should be a 1-D numpy array
    `n` should be an integer
    
    Returns a single-valued forecast for the next value in the series.
    """
    forecast = data[-n]
    return forecast


def fc_mean(data, n=3, **kwargs):
    """The 'mean' forecast of the next point in `data` (presumed to be 
    ordered in time) is equal to the mean value of the most recent `n` observed points.
    
    `data` should be a 1-D numpy array
    `n` should be an integer
    
    Returns a single-valued forecast for the next value in the series.
    """
    # don't start averaging until we've seen n points
    if len(data[-n:]) < n:
        forecast = np.nan
    else:
        # nb: we'll keep the forecast as a float
        forecast = np.mean(data[-n:])
    return forecast

def fc_drift(data, n=3, **kwargs):
    """The 'drift' forecast of the next point in `data` (presumed to be 
    ordered in time) is a linear extrapoloation from `n` points ago, through the
    most recent point.
    
    `data` should be a 1-D numpy array
    `n` should be an integer
    
    Returns a single-valued forecast for the next value in the series.
    """
    yi = data[-n]
    yf = data[-1]
    slope = (yf - yi) / (n-1)
    forecast = yf + slope
    return forecast


