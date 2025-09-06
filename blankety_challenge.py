import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.signal import savgol_filter
from flask import Blueprint, jsonify, request
import logging

logger = logging.getLogger(__name__)

blankety_bp = Blueprint('blankety', __name__)


def impute_series(series):
    """
    Impute missing values in a time series using multiple methods and choose the best approach.
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
    
    # Method 1: Cubic spline interpolation
    try:
        if len(valid_values) >= 4:
            cs = interpolate.CubicSpline(valid_indices, valid_values, extrapolate=True)
            imputed = cs(missing_indices)
            methods.append(('spline', imputed))
    except:
        pass
    
    # Method 2: Linear interpolation with extrapolation
    try:
        interp_func = interpolate.interp1d(valid_indices, valid_values, 
                                         kind='linear', fill_value='extrapolate')
        imputed = interp_func(missing_indices)
        methods.append(('linear', imputed))
    except:
        pass
    
    # Method 3: Polynomial fitting (try different degrees)
    for degree in [1, 2, 3]:
        try:
            if len(valid_values) > degree:
                coeffs = np.polyfit(valid_indices, valid_values, degree)
                poly = np.poly1d(coeffs)
                imputed = poly(missing_indices)
                methods.append((f'poly_{degree}', imputed))
        except:
            continue
    
    # Method 4: Savitzky-Golay filter-based approach
    try:
        if len(valid_values) >= 5:
            # Create a temporary complete series using linear interpolation
            temp_data = data.copy()
            interp_func = interpolate.interp1d(valid_indices, valid_values, 
                                             kind='linear', fill_value='extrapolate')
            temp_data[missing_mask] = interp_func(missing_indices)
            
            # Apply Savgol filter
            window_length = min(51, len(temp_data) if len(temp_data) % 2 == 1 else len(temp_data) - 1)
            if window_length < 5:
                window_length = 5
            if window_length % 2 == 0:
                window_length += 1
                
            smoothed = savgol_filter(temp_data, window_length, 3)
            imputed = smoothed[missing_indices]
            methods.append(('savgol', imputed))
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
    # (least variance in second derivatives for smoothness)
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
            
            # Calculate smoothness score (second derivative variance)
            if len(temp_data) >= 3:
                second_diff = np.diff(temp_data, n=2)
                score = np.var(second_diff) + 0.1 * np.var(imputed_vals)  # Penalize high variance
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