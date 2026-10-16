# Data Prep Pipeline:

Removes completely empty spectral bands.
Sorts spectral bands by wavelength.
Interpolates missing spectral values row by row.
Handles remaining invalid or missing numeric values.
Preserves Crop and Stage labels.
Removes identifier columns.
Generates the selected Milestone 2 features.
Adds spectral statistics, vegetation indices, red-edge features, gradient features, and temporal/geographical encodings.
Produces a fully numeric feature matrix apart from the target columns.
-> Generates the EDA plots successfully.



Current implemenation favours:

Physically meaningful wavelengths
Features that are interpretable
Features that are robust to missing spectral regions
Gradients calculated per nanometer
Avoidance of redundant feature combinations