.. _visualization:

Visualisation & Post-processing
================================

LilyTorch writes three kinds of output during a run:

* **PNG frames** of 2-D slices and 3-D isosurfaces, into
  ``<save_path>/<timestamp>/<field>/``.
* **HDF5** velocity / pressure fields (when ``save_uv: true``).
* **VTK** rectilinear grids (when ``save_vtk: true``) for ParaView.

This page covers the tools that turn those outputs into movies and
in-viewer overlays.

.. contents:: On this page
   :local:
   :depth: 2


Frames → video (``video_postprocess``)
--------------------------------------

After a run, assemble the saved PNG frames into MP4 or GIF:

.. code-block:: bash

   # Generate videos for ALL field sub-folders in a specific run
   python lilytorch/src/video_postprocess.py /path/to/save_path/2026-03-05_12-00-00/

   # Point at the parent save_path — the script picks the latest timestamped run
   python lilytorch/src/video_postprocess.py /path/to/save_path/

   # Only render selected fields, as animated GIFs
   python lilytorch/src/video_postprocess.py /path/to/run_dir \
       --fields omega_z_3d vel_mag_3d --format gif

Common 2-D field names: ``omega_z``, ``vel_mag``, ``pressure``.
Common 3-D field names: ``omega_x_3d``, ``omega_y_3d``, ``omega_z_3d``,
``omega_mag_3d``, ``vel_mag_3d``, ``pressure_3d``.

Options
^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Flag
     - Default
     - Description
   * - ``--fields F1 F2 …``
     - all
     - Only render the listed sub-folders.
   * - ``--fps N``
     - auto
     - Override frame rate (by default derived from ``dt`` and ``save_every``).
   * - ``--slow-factor S``
     - 1.0
     - Real-time multiplier; 1.0 means physical time equals video time.
   * - ``--no-overlay``
     - off
     - Disable the simulation-time text overlay on each frame.
   * - ``--crf N``
     - 18
     - H.264 quality — lower is better (18 ≈ visually lossless).
   * - ``--format {mp4,gif}``
     - mp4
     - Output format.

Notes:

* The FPS is inferred from ``parameters.yaml`` (``dt``, ``save_every``).
  If that file is missing the script defaults to 10 FPS.
* ``ffmpeg`` is used when available; otherwise the script falls back to
  OpenCV's ``VideoWriter`` (MP4) or Pillow (GIF).
* Outputs are written alongside the PNG sub-folders (e.g.
  ``omega_z_3d.mp4``).


Top-view PIV-style curl (``projected_field_postprocess``)
---------------------------------------------------------

For matching an experimental top camera, lilytorch also ships a
dedicated tool that reads ``u``, ``v``, ``w`` from ``fields.h5``,
collapses the in-plane velocity through a user-selected depth slab,
computes the 2-D curl on the projected field, and encodes the result as
PNG + MP4 / GIF.

.. code-block:: bash

   python lilytorch/src/projected_field_postprocess.py /path/to/run_dir \
       --camera-axis -z \
       --projection-mode gaussian \
       --zlim -0.04 0.04 \
       --focus-depth 0.0 \
       --depth-sigma 0.015 \
       --xlim -0.8 0.6 \
       --ylim -0.25 0.25 \
       --output-tag top_piv

Key parameters
^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Flag
     - Meaning
   * - ``--camera-axis``
     - Orthographic viewing direction (``-z`` is the standard top view).
   * - ``--xlim``, ``--ylim``
     - Camera window in the image plane.
   * - ``--zlim``
     - Depth slab included in the projection.
   * - ``--projection-mode``
     - How the depth slab is collapsed: ``mean``, ``sum``, ``max_abs``,
       ``gaussian``.
   * - ``--coarsen``
     - Block-average the projected velocity before curl (mimics a larger
       PIV interrogation window).
   * - ``--velocity-sigma``
     - Gaussian smoothing on the projected velocity before curl — the
       main control for suppressing tiny vortices.
   * - ``--curl-sigma``
     - Optional extra smoothing on the final curl image.
   * - ``--focus-depth``, ``--depth-sigma``
     - Depth emphasis used by the Gaussian projection mode.
   * - ``--camera-roll``
     - In-plane image rotation after projection.
   * - ``--flip-horizontal``, ``--flip-vertical``
     - Image mirroring, to match camera orientation.
   * - ``--output-tag``
     - Suffix for the generated frame folder and video.
   * - ``--no-video``
     - Keep PNG frames only, skip the movie encoding.

.. tip::

   If the synthetic movie shows many more small-scale vortices than the
   physical PIV, first increase ``--velocity-sigma`` and then
   ``--coarsen`` — real PIV returns an interrogation-window average
   before vorticity is computed.


In-viewer flow (``FlowViewer``, coupled mode only)
--------------------------------------------------

:class:`~lilytorch.integration.flow_viewer.FlowViewer` is a FARMS
``TaskExtension`` that draws the fluid field as coloured spheres
*inside* the MuJoCo viewer window **and** in the recorded camera video,
overlaid on the swimming animat. This is only relevant for coupled
runs (Path B of :doc:`getting_started`).

Requirements
^^^^^^^^^^^^

* A 3-D fluid solver must be active — ``FlowViewer`` is skipped for 2-D runs.
* ``FlowViewer`` must be listed **after** ``FluidExtension`` in the
  extensions list; it reads state that ``FluidExtension`` updates in
  ``before_step``.
* For the interactive GUI: ``headless: False``.
* For the recorded video: include a ``CameraRecording`` extension in
  the extensions list. ``FlowViewer`` automatically patches its
  offscreen renderer. Both modes work simultaneously.

How it works
^^^^^^^^^^^^

At every ``update_every`` timesteps (default: the solver's
``save_every``) ``FlowViewer`` extracts the chosen scalar field from
the solver, Gaussian-smooths it, crops boundary cells, masks out the
body interior, thresholds at
``iso_fraction × peak|field|``, subsamples down to the sphere budget,
and updates sphere positions + colours. Bipolar fields (vorticity
components, pressure) use red for positive and blue for negative;
non-negative fields (``vel_mag``, ``omega_mag``) use orange.

Configuration
^^^^^^^^^^^^^

.. code-block:: python

   extensions = [
       # ... FluidExtension must come first ...
       {
           "loader": "lilytorch.integration.flow_viewer.FlowViewer",
           "config": {
               "field":         "omega_z",   # scalar field to display
               "max_spheres":   4000,         # sphere budget
               "iso_fraction":  0.15,         # threshold = fraction × peak |field|
               "smooth_sigma":  2.5,          # Gaussian σ (grid cells)
               "crop_boundary": 3,            # cells cropped from each face
               "sphere_size":   0.004,        # sphere radius [MuJoCo units]
               "update_every":  None,         # None → solver.save_every
           },
       },
   ]

Parameters
^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - Parameter
     - Default
     - Description
   * - ``field``
     - ``"omega_z"``
     - Scalar field to visualise. One of ``omega_x``, ``omega_y``,
       ``omega_z``, ``omega_mag``, ``vel_mag``, ``pressure``.
   * - ``max_spheres``
     - ``4000``
     - Max number of sphere geoms. MuJoCo's hard limit is 100 000 per
       scene; higher values give denser coverage at more rendering cost.
   * - ``iso_fraction``
     - ``0.15``
     - Isosurface threshold as a fraction of the peak absolute value.
       Lower values show more of the field; higher values isolate the
       strongest features.
   * - ``smooth_sigma``
     - ``2.5``
     - Gaussian σ (in cells) applied before thresholding. ``0`` disables.
   * - ``crop_boundary``
     - ``3``
     - Number of cells removed from each face of the domain.
   * - ``sphere_size``
     - ``0.004``
     - Sphere radius in MuJoCo world units.
   * - ``update_every``
     - ``None``
     - Refresh cadence in solver iterations; ``None`` → ``solver.save_every``.

Available fields
^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 22 48 30

   * - Field
     - Description
     - Colour scheme
   * - ``omega_x``
     - Vorticity x-component
     - Bipolar (red +, blue −)
   * - ``omega_y``
     - Vorticity y-component
     - Bipolar
   * - ``omega_z``
     - Vorticity z-component
     - Bipolar
   * - ``omega_mag``
     - Vorticity magnitude
     - Orange
   * - ``vel_mag``
     - Velocity magnitude
     - Orange
   * - ``pressure``
     - Pressure field
     - Bipolar

Tuning tips
^^^^^^^^^^^

* **Too few / too many spheres?** Adjust ``iso_fraction``. ``0.10`` shows
  more of the wake; ``0.25`` highlights only the strongest vortices.
* **Speckled appearance?** Increase ``smooth_sigma`` (e.g. 3.0–4.0).
* **Spheres too small or too large?** Scale ``sphere_size`` relative to
  your swimmer's body length.
* **Slow rendering?** Reduce ``max_spheres`` or raise ``update_every``.
* **Want pressure instead of vorticity?** Set ``field`` to ``"pressure"``
  or ``"vel_mag"``.
