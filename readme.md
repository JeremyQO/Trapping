# Trapping
* The file `ffd_mode_solver.py` contains a class that can be used in order to calculate the eigenmodes of a straight waveguide, given a description of the (potentially) anisotropic and inhomogeneous index of refraction in the transverse section of the waveguide. 
* The file `data_structure.py` contains the classes used in order to import or generate data about the dipole beams. The data can either be imported from csv files containing the result of COMSOL simulations, or be calculated on the spot using the class described above
* The file `trapping.py` contains a class that computes the potential seen by an atom in the presence of an optical electric field. To do this, one needs to define the fields using the classes from `data_structure.py`. 
