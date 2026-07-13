import xarray as xr

ds = xr.open_dataset('data/asc/chimanimani_63.nc')
print(ds)