
RUN:
- Run a full self-propelled 3d 1guilla simulation to check if the force computation in 3d works
- Run error analysis for the flow past cylinder in 2d following run_error_analysis_cylinder.py (adapting this script)

TO DO LIST:
1. Test a simulation with an analytically moving body, both analytically defined and defined via a mesh file.
2. I want to understand if it is possible to make a unique shared BDIMhandler for all mujoco simulations (if possible - check that they can share it) 2d and 3d simulations. First check all BDIMhandler files and see how they differ. Clarify if it is possible to define hyperparameters that are generated via the config generation files instead of the BDIMhandler. This would simplified greatly the repo.
3. Make a new 2d/3d salamander simulation for the salamander and zebrafish models
4. Run 2d sphere coquerelle and gazzola tests again
5. The pool appears above the swimmer position. I want the swimmer to be inside the pool. Also, the water arena should be programmatically be generated to have water inside the pool only. Also the sizes of the pool borders should be scaled with the pool size - make it automatically generate them from the generation files in the config run. It would be nice to have some cool textures for the pool as well.
6. Still need to i
7. Polish the repository


