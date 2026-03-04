"""
PreCICE handler for coupling FARMS MuJoCo simulation with OpenFOAM.

This replaces the BDIM handler when use_precice=True.  At each timestep:
  1. Extract link surface mesh positions and velocities from FARMS data.
  2. Compute displacement of every mesh vertex relative to the reference pose.
  3. Write displacements to preCICE → OpenFOAM moves its mesh.
  4. Read forces from preCICE ← OpenFOAM computes fluid forces.
  5. Apply forces to MuJoCo bodies via xfrc_applied.
"""

import os
import numpy as np
import precice

import logging
logger = logging.getLogger(__name__)


class PreCICEHandler:
    """Couples FARMS articulated-body simulation with OpenFOAM via preCICE."""

    def __init__(self, config, data, physics):
        """
        Parameters
        ----------
        config : dict
            Configuration dictionary with keys:
              - precice_config : str, path to precice-config.xml
              - stl_folder     : str, path to folder with link*_collision.stl files
              - link_names     : list[str]
              - n_links        : int
        data : list
            List of FARMS AnimatData objects (one per animat).
        physics : dm_control Physics
            MuJoCo physics handle.
        """
        self.data = data
        self.iteration = 0
        self.terminate = False
        self.config = config
        self._ref_initialized = False  # Will be set on first step()

        precice_config = config.get("precice_config", "precice-config.xml")
        stl_folder = config.get("stl_folder", "")
        self.link_names = config.get("link_names", [])
        self.n_links = len(self.link_names)
        self.rho_fluid = config.get("rho_fluid", 1000.0)
        # MuJoCo prefixes body names with a{animat_i}_
        self.animat_index = config.get("animat_index", 0)
        self.body_prefix = f"a{self.animat_index}_"

        # ---- Load reference STL vertices for each link ----
        self._load_stl_meshes(stl_folder)

        # ---- Transform STL vertices to world frame using initial MuJoCo poses ----
        self._transform_to_world_frame(physics)

        # ---- Initialize preCICE ----
        self.participant = precice.Participant(
            "FARMS",
            precice_config,
            0,  # rank
            1,  # size
        )

        mesh_name = "FARMS-Mesh"

        # Combine all link vertices into a single coupling mesh (world frame)
        self.all_vertices = np.vstack(self.ref_vertices_world)  # (N_total, 3)
        self.n_vertices = self.all_vertices.shape[0]

        # Track which vertices belong to which link
        self.link_vertex_ranges = []
        offset = 0
        for verts in self.ref_vertices:
            n = verts.shape[0]
            self.link_vertex_ranges.append((offset, offset + n))
            offset += n

        # Reference positions in world frame (initial MuJoCo pose)
        self.ref_positions = self.all_vertices.copy()

        # Current displaced positions
        self.current_positions = self.all_vertices.copy()

        # Set mesh vertices in preCICE (world frame for correct mapping)
        self.vertex_ids = self.participant.set_mesh_vertices(
            mesh_name,
            self.all_vertices,
        )

        # Data IDs
        self.mesh_name = mesh_name

        # Initialize storage for forces
        self.forces = np.zeros((self.n_vertices, 3))

        # Displacement relative to reference
        self.displacements = np.zeros((self.n_vertices, 3))

        # Initialize preCICE
        self.participant.initialize()

        self.dt_precice = self.participant.get_max_time_step_size()

        logger.info(
            f"PreCICE initialized: {self.n_vertices} vertices, "
            f"dt_precice={self.dt_precice}"
        )

    def _load_stl_meshes(self, stl_folder):
        """Load STL collision meshes for each link and extract unique vertices."""
        try:
            from stl import mesh as stl_mesh
        except ImportError:
            raise ImportError(
                "numpy-stl is required for PreCICE coupling. "
                "Install with: pip install numpy-stl"
            )

        self.ref_vertices = []
        self.ref_normals = []

        for link_name in self.link_names:
            stl_path = os.path.join(stl_folder, f"{link_name}_collision.stl")
            if not os.path.exists(stl_path):
                logger.warning(f"STL file not found: {stl_path}, using empty mesh")
                self.ref_vertices.append(np.zeros((0, 3)))
                self.ref_normals.append(np.zeros((0, 3)))
                continue

            m = stl_mesh.Mesh.from_file(stl_path)

            # Get unique vertices from the triangulated surface
            all_verts = m.vectors.reshape(-1, 3)
            unique_verts = np.unique(all_verts, axis=0)

            # Compute face normals for later force distribution
            normals = m.normals

            self.ref_vertices.append(unique_verts)
            self.ref_normals.append(normals)

            logger.info(
                f"  {link_name}: {unique_verts.shape[0]} unique vertices "
                f"from {m.vectors.shape[0]} triangles"
            )

    def _transform_to_world_frame(self, physics):
        """Transform STL vertices from link-local frame to world frame
        using the initial MuJoCo body poses."""
        self.ref_vertices_world = []

        for link_id, link_name in enumerate(self.link_names):
            local_verts = self.ref_vertices[link_id]
            if local_verts.shape[0] == 0:
                self.ref_vertices_world.append(local_verts)
                continue

            try:
                # Get initial body pose from MuJoCo (bodies are prefixed with a{i}_)
                mujoco_name = f"{self.body_prefix}{link_name}"
                body_xpos = np.array(physics.named.data.xpos[mujoco_name])
                body_xmat = np.array(physics.named.data.xmat[mujoco_name]).reshape(3, 3)

                world_verts = (body_xmat @ local_verts.T).T + body_xpos
                self.ref_vertices_world.append(world_verts)

                logger.info(
                    f"  {link_name}: transformed to world frame, "
                    f"body_pos={body_xpos}, "
                    f"verts range x=[{world_verts[:,0].min():.4f}, {world_verts[:,0].max():.4f}], "
                    f"y=[{world_verts[:,1].min():.4f}, {world_verts[:,1].max():.4f}], "
                    f"z=[{world_verts[:,2].min():.4f}, {world_verts[:,2].max():.4f}]"
                )
            except Exception as e:
                logger.warning(
                    f"Could not get MuJoCo pose for {link_name}: {e}. "
                    f"Using local-frame vertices."
                )
                self.ref_vertices_world.append(local_verts)

    def _update_vertex_positions(self, iteration, physics):
        """
        Transform reference STL vertices to current world-frame positions
        using the LIVE MuJoCo body poses (physics.data.xpos / xmat).

        This reads the actual current state of the bodies, which is critical
        for implicit coupling where the handler may be called multiple times
        at the same logical iteration (sub-iterations) and checkpoint restores
        modify physics state directly.
        """
        for link_id, link_name in enumerate(self.link_names):
            start, end = self.link_vertex_ranges[link_id]
            if start == end:
                continue

            # Read current body pose directly from MuJoCo
            mujoco_name = f"{self.body_prefix}{link_name}"
            body_xpos = np.array(physics.named.data.xpos[mujoco_name])
            body_xmat = np.array(physics.named.data.xmat[mujoco_name]).reshape(3, 3)

            # Transform local STL vertices to current world frame
            local_verts = self.ref_vertices[link_id]  # (n, 3)
            world_verts = (body_xmat @ local_verts.T).T + body_xpos

            self.current_positions[start:end] = world_verts

        # On the first call, initialize ref_positions to world-frame t=0 positions
        # so that displacement = 0 on the first timestep
        if not self._ref_initialized:
            self.ref_positions = self.current_positions.copy()
            self._ref_initialized = True
            logger.info(
                f"Initialized ref_positions from world-frame poses at iteration {iteration}"
            )

        # Displacement = current - reference (both in world frame now)
        self.displacements = self.current_positions - self.ref_positions

        # Debug: log displacement statistics
        disp_mag = np.linalg.norm(self.displacements, axis=1)
        logger.info(
            f"Displacement stats: min={disp_mag.min():.6e}, max={disp_mag.max():.6e}, "
            f"mean={disp_mag.mean():.6e}, "
            f"dx=[{self.displacements[:,0].min():.6e}, {self.displacements[:,0].max():.6e}], "
            f"dy=[{self.displacements[:,1].min():.6e}, {self.displacements[:,1].max():.6e}], "
            f"dz=[{self.displacements[:,2].min():.6e}, {self.displacements[:,2].max():.6e}]"
        )

    def apply_forces(self, task, physics):
        """Apply preCICE forces back to MuJoCo bodies."""
        for animat_data_idx, animat_data in enumerate(self.data):
            for link_id, link_name in enumerate(self.link_names):
                start, end = self.link_vertex_ranges[link_id]
                if start == end:
                    continue

                # Sum all vertex forces for this link
                link_forces = self.forces[start:end]  # (n_verts, 3)
                total_force = link_forces.sum(axis=0)  # (3,)

                # Compute torque about COM
                com_pos = np.array(
                    animat_data.sensors.links.com_positions()[self.iteration, link_id]
                )
                r = self.current_positions[start:end] - com_pos  # (n, 3)
                torques = np.cross(r, link_forces)  # (n, 3)
                total_torque = torques.sum(axis=0)  # (3,)

                # Apply to MuJoCo xfrc_applied [fx, fy, fz, tx, ty, tz]
                ind = task.maps[animat_data_idx]['sensors']['data2xfrc'][link_id]
                physics.data.xfrc_applied[ind, 0] = total_force[0] * task.units.newtons
                physics.data.xfrc_applied[ind, 1] = total_force[1] * task.units.newtons
                physics.data.xfrc_applied[ind, 2] = total_force[2] * task.units.newtons
                physics.data.xfrc_applied[ind, 3] = total_torque[0] * task.units.newtons
                physics.data.xfrc_applied[ind, 4] = total_torque[1] * task.units.newtons
                physics.data.xfrc_applied[ind, 5] = total_torque[2] * task.units.newtons

    def step(self, task, physics):
        """
        Perform one preCICE coupling step with implicit iteration support:
          1. Save checkpoint if preCICE requires it
          2. Update mesh vertex positions from FARMS
          3. Write displacements to preCICE
          4. Advance preCICE (triggers OpenFOAM fluid solve)
          5. Read forces from preCICE
          6. Reload checkpoint if preCICE requires it (iteration not converged)
          7. Apply forces to MuJoCo bodies
        """
        if self.terminate or not self.participant.is_coupling_ongoing():
            self.terminate = True
            return

        iteration = self.iteration

        # --- Implicit coupling: save checkpoint ---
        if self.participant.requires_writing_checkpoint():
            self._save_checkpoint(physics)

        # 1. Update vertex positions from current MuJoCo state
        self._update_vertex_positions(iteration, physics)

        # 2. Write displacement data to preCICE
        self.participant.write_data(
            self.mesh_name,
            "Displacement",
            self.vertex_ids,
            self.displacements,
        )

        # 3. Advance preCICE
        self.participant.advance(self.dt_precice)
        self.dt_precice = self.participant.get_max_time_step_size()

        # --- Implicit coupling: reload checkpoint if not converged ---
        if self.participant.requires_reading_checkpoint():
            self._load_checkpoint(physics)
        else:
            # Converged — this time window is done, move to next iteration
            self.iteration += 1

        # 4. Read force data from preCICE
        self.forces = self.participant.read_data(
            self.mesh_name,
            "Force",
            self.vertex_ids,
            self.dt_precice,
        )

        # 5. Apply forces to MuJoCo (DISABLED — one-way coupling test)
        # self.apply_forces(task, physics)

        if not self.participant.is_coupling_ongoing():
            self.terminate = True
            self.participant.finalize()
            logger.info("PreCICE coupling finalized.")

    def _save_checkpoint(self, physics):
        """Save MuJoCo state for implicit coupling rollback."""
        self._checkpoint_qpos = physics.data.qpos.copy()
        self._checkpoint_qvel = physics.data.qvel.copy()
        self._checkpoint_positions = self.current_positions.copy()
        self._checkpoint_iteration = self.iteration
        logger.debug(f"Saved checkpoint at iteration {self.iteration}")

    def _load_checkpoint(self, physics):
        """Restore MuJoCo state for implicit coupling rollback."""
        physics.data.qpos[:] = self._checkpoint_qpos
        physics.data.qvel[:] = self._checkpoint_qvel
        self.current_positions[:] = self._checkpoint_positions
        self.iteration = self._checkpoint_iteration
        logger.debug(f"Loaded checkpoint at iteration {self.iteration}")

    def finalize(self):
        """Clean up preCICE."""
        if self.participant.is_coupling_ongoing():
            self.participant.finalize()
