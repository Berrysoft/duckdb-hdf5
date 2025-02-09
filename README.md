# DuckDB HDF5 extension
This is a community extension to read HDF5 files.

## The `read_hdf5` function
It reads a dataset from an HDF5 file.
```sql
FROM read_hdf5("example_file.h5", "dataset_name");
```
