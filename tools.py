# %%
import xarray as xr
import numpy as np

# %%
def verify_datasets(ds_baseline, ds_new, rtol=1e-5, atol=1e-8):
    """
    Compares variables in two xarray Datasets within specified tolerances.
    
    rtol: Relative tolerance (e.g., 1e-5 is 0.001% difference)
    atol: Absolute tolerance (useful for variables close to zero)
    """
    all_match = True
    
    # 1. Check if the datasets contain the exact same variables
    vars_base = set(ds_baseline.data_vars)
    vars_new = set(ds_new.data_vars)
    
    if vars_base != vars_new:
        print("⚠️  Warning: Datasets contain different variables!")
        print(f"   Missing in new: {vars_base - vars_new}")
        print(f"   Extra in new: {vars_new - vars_base}")
        return False

    # 2. Setup the summary table
    print(f"{'Variable':<20} | {'Status':<8} | {'Max Abs Diff':<14} | {'Max Rel Diff':<14}")
    print("-" * 65)

    # 3. Loop through and compare each variable
    for var in sorted(vars_base):
        # Extract underlying numpy arrays for speed
        arr_base = ds_baseline[var].values
        arr_new = ds_new[var].values
        
        # Check if the arrays have the same shape
        if arr_base.shape != arr_new.shape:
            print(f"{var:<20} | ❌ SHAPE  | Mismatch: {arr_base.shape} vs {arr_new.shape}")
            all_match = False
            continue

        # Use np.allclose to check within tolerance, treating NaNs as equal
        is_close = np.allclose(arr_base, arr_new, rtol=rtol, atol=atol, equal_nan=True)
        
        # Calculate maximum differences for reporting
        # Mask out NaNs to avoid warnings during subtraction/division
        mask = ~np.isnan(arr_base) & ~np.isnan(arr_new)
        
        if np.any(mask):
            diff = np.abs(arr_base[mask] - arr_new[mask])
            max_abs = np.max(diff)
            
            # Suppress divide-by-zero warnings for relative difference
            with np.errstate(divide='ignore', invalid='ignore'):
                rel_diff = diff / np.abs(arr_base[mask])
                # If arr_base was 0 and diff was 0, rel_diff is NaN. Convert to 0.
                rel_diff = np.nan_to_num(rel_diff, nan=0.0, posinf=0.0, neginf=0.0)
                max_rel = np.max(rel_diff)
        else:
            max_abs = 0.0
            max_rel = 0.0
            
        # Check for NaN mask mismatches
        if not np.array_equal(np.isnan(arr_base), np.isnan(arr_new)):
            print(f"{var:<20} | ❌ MASK   | NaN patterns do not match!")
            all_match = False
            continue

        # Format output
        status = "✅ PASS" if is_close else "❌ FAIL"
        if not is_close:
            all_match = False
            
        print(f"{var:<20} | {status:<8} | {max_abs:<14.3e} | {max_rel:<14.3e}")

    print("-" * 65)
    
    if all_match:
        print("🎉 SUCCESS: All variables match within the specified tolerances.")
    else:
        print("⚠️  FAILURE: Some variables diverge beyond the acceptable thresholds.")
        
    return all_match
