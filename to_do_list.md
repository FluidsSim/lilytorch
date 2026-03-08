
RUN:
- Run a full self-propelled 3d 1guilla simulation to check if the force computation in 3d works
- Run error analysis for the flow past cylinder in 2d following run_error_analysis_cylinder.py (adapting this script)
- The pool appears above the swimmer position. I want the swimmer to be inside the pool. Also, the water arena should be programmatically be generated to have water inside the pool only. Also the sizes of the pool borders should be scaled with the pool size - make it automatically generate them from the generation files in the config run. It would be nice to have some cool textures for the pool as well.

- Add a proper analusis of the computational cost of the main parts of the code for 1guilla 3d swimming, for example what is the cost of the different parts of the solver, the computation of the sdf properties, the posson solver, the advection, etc. Suggest a small list of cost to tests
- Move the yaml files in a dedicated folder. Fix the yaml files to adhere to the most recent
- I want to understand if it is possible to make a unique shared BDIMhandler for all mujoco simulations (if possible - check that they can share it) 2d and 3d simulations. First check all BDIMhandler files and see how they differ. Clarify if it is possible to define hyperparameters that are generated via the config generation files instead of the BDIMhandler. This would simplified greatly the repo.



HIGH PRIORITY:

- Test a simulation with an analytically moving body, both analytically defined and defined via a mesh file.
- Make a new 2d/3d salamander simulation for the salamander and zebrafish models
- Run 2d sphere coquerelle and gazzola tests again
- Polish the repository
- Run the drag cylinder test



LOW PRIORITY:
- Implement faster solver for the Poisson equation, i.e. test different smoother, or preconjugate gradient multigrid method 


LONG TERM GOALS:
- Add sph simulation support
- Add volume of fluids methods for handling water surface breaking
- Monolithic fluid multi rigid body solver (?)



