import numpy as np
from flask import Blueprint, jsonify, request
import logging

logger = logging.getLogger(__name__)

blankety_bp = Blueprint('blankety', __name__)


def cubic_spline_coeffs(x, y):
    """Simple cubic spline implementation using numpy only"""
    n = len(x)
    if n < 2:
        return None
    
    # Create system of equations for natural cubic spline
    h = np.diff(x)
    alpha = np.zeros(n-1)
    
    for i in range(1, n-1):
        alpha[i] = 3/h[i] * (y[i+1] - y[i]) - 3/h[i-1] * (y[i] - y[i-1])
    
    # Solve tridiagonal system
    l = np.ones(n)
    mu = np.zeros(n)
    z = np.zeros(n)
    
    for i in range(1, n-1):
        l[i] = 2 * (x[i+1] - x[i-1]) - h[i-1] * mu[i-1]
        if abs(l[i]) < 1e-10:
            return None
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i-1] * z[i-1]) / l[i]
    
    # Back substitution
    c = np.zeros(n)
    b = np.zeros(n-1)
    d = np.zeros(n-1)
    
    for j in range(n-2, -1, -1):
        c[j] = z[j] - mu[j] * c[j+1]
        b[j] = (y[j+1] - y[j]) / h[j] - h[j] * (c[j+1] + 2*c[j]) / 3
        d[j] = (c[j+1] - c[j]) / (3 * h[j])
    
    return {'a': y[:-1], 'b': b, 'c': c[:-1], 'd': d, 'x': x[:-1]}


def evaluate_spline(spline, x_new):
    """Evaluate cubic spline at new points"""
    if spline is None:
        return None
    
    result = np.zeros(len(x_new))
    for i, xi in enumerate(x_new):
        # Find the right interval
        j = np.searchsorted(spline['x'], xi) - 1
        j = max(0, min(j, len(spline['x']) - 1))
        
        dx = xi - spline['x'][j]
        result[i] = (spline['a'][j] + spline['b'][j] * dx + 
                    spline['c'][j] * dx**2 + spline['d'][j] * dx**3)
    
    return result


def moving_average_smooth(data, window=5):
    """Simple moving average smoothing"""
    if len(data) < window:
        return data
    
    smoothed = np.copy(data)
    half_window = window // 2
    
    for i in range(half_window, len(data) - half_window):
        smoothed[i] = np.mean(data[i-half_window:i+half_window+1])
    
    return smoothed


def impute_series(series):
    """
    Impute missing values in a time series using numpy-only methods.
    """
    # Convert to numpy array and find missing indices
    data = np.array(series, dtype=float)
    missing_mask = np.isnan(data)
    valid_mask = ~missing_mask
    
    if not np.any(missing_mask):
        return series  # No missing values
    
    if not np.any(valid_mask):
        return [0.0] * len(series)  # All missing, return zeros
    
    # Get valid indices and values
    valid_indices = np.where(valid_mask)[0]
    valid_values = data[valid_mask]
    missing_indices = np.where(missing_mask)[0]
    
    if len(valid_values) < 2:
        # Too few points for interpolation, use mean
        mean_val = np.mean(valid_values)
        data[missing_mask] = mean_val
        return data.tolist()
    
    # Try multiple imputation methods
    methods = []
    
    # Method 1: Linear interpolation with extrapolation
    try:
        imputed = np.interp(missing_indices, valid_indices, valid_values)
        methods.append(('linear', imputed))
    except:
        pass
    
    # Method 2: Polynomial fitting (try different degrees)
    for degree in [1, 2, 3]:
        try:
            if len(valid_values) > degree:
                coeffs = np.polyfit(valid_indices, valid_values, degree)
                poly = np.poly1d(coeffs)
                imputed = poly(missing_indices)
                # Clip extreme values
                if len(valid_values) > 0:
                    val_range = np.max(valid_values) - np.min(valid_values)
                    val_mean = np.mean(valid_values)
                    imputed = np.clip(imputed, 
                                    val_mean - 3*val_range, 
                                    val_mean + 3*val_range)
                methods.append((f'poly_{degree}', imputed))
        except:
            continue
    
    # Method 3: Cubic spline (numpy implementation)
    try:
        if len(valid_values) >= 4:
            spline = cubic_spline_coeffs(valid_indices.astype(float), valid_values)
            if spline is not None:
                imputed = evaluate_spline(spline, missing_indices.astype(float))
                if imputed is not None and not np.any(np.isnan(imputed)):
                    methods.append(('spline', imputed))
    except:
        pass
    
    # Method 4: Moving average based interpolation
    try:
        if len(valid_values) >= 5:
            # First do linear interpolation
            temp_data = data.copy()
            temp_data[missing_mask] = np.interp(missing_indices, valid_indices, valid_values)
            
            # Apply moving average smoothing
            smoothed = moving_average_smooth(temp_data, window=min(11, len(temp_data)//10 + 1))
            imputed = smoothed[missing_indices]
            methods.append(('moving_avg', imputed))
    except:
        pass
    
    # Method 5: Local linear regression (simple version)
    try:
        imputed_vals = []
        for miss_idx in missing_indices:
            # Find nearby points for local fitting
            distances = np.abs(valid_indices - miss_idx)
            n_neighbors = min(10, len(valid_indices))
            nearest_idx = np.argsort(distances)[:n_neighbors]
            
            local_x = valid_indices[nearest_idx]
            local_y = valid_values[nearest_idx]
            
            if len(local_x) >= 2:
                # Simple linear regression
                coeffs = np.polyfit(local_x, local_y, 1)
                imputed_val = np.polyval(coeffs, miss_idx)
            else:
                imputed_val = np.mean(local_y)
            
            imputed_vals.append(imputed_val)
        
        methods.append(('local_linear', np.array(imputed_vals)))
    except:
        pass
    
    # If no methods worked, use simple linear interpolation
    if not methods:
        try:
            imputed = np.interp(missing_indices, valid_indices, valid_values)
            methods.append(('simple_interp', imputed))
        except:
            # Last resort: use mean
            mean_val = np.mean(valid_values)
            imputed = np.full(len(missing_indices), mean_val)
            methods.append(('mean', imputed))
    
    # Choose the method that produces the most stable results
    best_method = None
    best_score = float('inf')
    
    for method_name, imputed_vals in methods:
        try:
            # Create complete series
            temp_data = data.copy()
            temp_data[missing_mask] = imputed_vals
            
            # Check for invalid values
            if np.any(np.isnan(imputed_vals)) or np.any(np.isinf(imputed_vals)):
                continue
            
            # Calculate smoothness score (variance of second differences)
            if len(temp_data) >= 3:
                second_diff = np.diff(temp_data, n=2)
                smoothness_score = np.var(second_diff)
                # Also penalize extreme deviations from local trends
                deviation_score = 0.1 * np.var(imputed_vals)
                score = smoothness_score + deviation_score
            else:
                score = np.var(imputed_vals)
            
            if score < best_score:
                best_score = score
                best_method = imputed_vals
        except:
            continue
    
    # Apply the best imputation
    if best_method is not None:
        data[missing_mask] = best_method
    else:
        # Fallback to mean
        mean_val = np.mean(valid_values)
        data[missing_mask] = mean_val
    
    return data.tolist()


@blankety_bp.route('/blankety', methods=['POST'])
def blankety():
    try:
        # Get the input data
        input_data = request.get_json()
        
        if not input_data or 'series' not in input_data:
            return jsonify({'error': 'Missing series data'}), 400
        
        series_list = input_data['series']
        
        if len(series_list) != 100:
            return jsonify({'error': f'Expected 100 series, got {len(series_list)}'}), 400
        
        # Process each series
        imputed_series = []
        for i, series in enumerate(series_list):
            if len(series) != 1000:
                return jsonify({'error': f'Series {i} has {len(series)} elements, expected 1000'}), 400
            
            # Replace None with np.nan for processing
            series_with_nan = [np.nan if x is None else x for x in series]
            imputed = impute_series(series_with_nan)
            imputed_series.append(imputed)
        
        return jsonify({'answer': imputed_series})
        
    except Exception as e:
        logger.error(f"Error in blankety endpoint: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500