"""CameraRecording that also dumps each rendered frame as a PNG.

``CameraRecordingFrames`` subclasses
``farms_mujoco.sensors.camera.CameraRecording`` and adds per-frame PNG export.
Because it reuses the parent's render pass verbatim, the PNG frames are exactly
the frames that compose the parent's video (``video.mp4``, ``video_follow.mp4``,
or any other camera you configure) — same camera, same resolution, same timing.

Usage
-----
Take any existing ``CameraRecording`` extension entry and swap its ``loader`` to
this class. That is the whole "pick which camera to dump frames for" mechanism::

    extensions.append({
        "loader": "lilytorch.integration.camera_frame_recorder.CameraRecordingFrames",
        "config": {
            "path"      : os.path.join(output_folder, "output", "video_follow.mp4"),
            "animat_id" : 0,
            "fps"       : 30,
            # ... the rest of the camera config, unchanged ...
            # optional extras handled by this subclass:
            "frames_dir"     : os.path.join(output_folder, "output", "video_follow_frames"),
            "save_video"     : True,   # keep writing the .mp4 too (default True)
            "png_compression": 3,      # PNG compression level 0-9 (default 3)
        },
    })

If ``frames_dir`` is omitted it defaults to ``<video_path>_frames`` (e.g.
``.../video_follow.mp4`` -> ``.../video_follow_frames/``). Frames are named
``frame_%06d.png`` with a sequential counter, ready for::

    ffmpeg -framerate <fps> -i frame_%06d.png -pix_fmt yuv420p out.mp4

Requires OpenCV (``cv2``); without it, PNG export is skipped with a warning.
"""

from __future__ import annotations

import os

from farms_core import pylog
from farms_mujoco.sensors.camera import (
    CameraRecording,
    CameraRecordingOptions,
    USE_CV2,
)

if USE_CV2:
    import cv2


class CameraRecordingFrames(CameraRecording):
    """A CameraRecording that additionally writes each frame as a PNG."""

    @classmethod
    def from_options(cls, config: dict, experiment_options):
        """From options.

        Pops this subclass's extra keys before handing the remaining (standard
        camera) keys to ``CameraRecordingOptions``, which rejects unknown keys.
        """
        config = dict(config)  # don't mutate caller's dict
        frames_dir = config.pop('frames_dir', None)
        save_video = config.pop('save_video', True)
        png_compression = config.pop('png_compression', 3)

        sim = experiment_options.simulation
        obj = cls(
            timestep=sim.physics.timestep,
            n_iterations=sim.runtime.n_iterations,
            viewer=sim.mujoco.viewer,
            **CameraRecordingOptions(**config),
        )
        obj._frames_dir = frames_dir or f'{obj.video.path}_frames'
        obj._save_video = bool(save_video)
        obj._png_compression = int(png_compression)
        obj._png_enabled = USE_CV2
        return obj

    def initialize_episode(self, task, physics):
        """Set up the parent recorder, then the frames directory."""
        super().initialize_episode(task, physics)

        if not self._png_enabled:
            pylog.warning(
                'CameraRecordingFrames: OpenCV (cv2) unavailable, PNG export off'
            )

        # Drop the video if the user only wants frames.
        if not self._save_video and self.out is not None:
            self.out.release()
            self.out = None
            video_file = f'{self.video.path}{self.video.file_extension}'
            try:
                os.remove(video_file)
            except OSError:
                pass

        if self._png_enabled:
            os.makedirs(self._frames_dir, exist_ok=True)
            pylog.info(
                'CameraRecordingFrames -> %s (save_video=%s)',
                self._frames_dir, self._save_video,
            )

    def before_step(self, task, action, physics):
        """Render via the parent, then persist the captured frame as a PNG."""
        prev_sample = self.sample
        super().before_step(task, action, physics)
        if not self._png_enabled:
            return
        if self.sample <= prev_sample:
            return  # no frame captured this iteration
        idx = self.sample - 1
        # self.data holds RGB; cv2.imwrite expects BGR.
        bgr = cv2.cvtColor(self.data[idx], cv2.COLOR_RGB2BGR)
        fname = os.path.join(self._frames_dir, f'frame_{idx:06d}.png')
        cv2.imwrite(fname, bgr, [cv2.IMWRITE_PNG_COMPRESSION, self._png_compression])

    def end_episode(self, task, physics):
        """Finalise the video (if kept) and report the PNG count."""
        if self._save_video:
            super().end_episode(task, physics)
        else:
            # Skip the parent's video save() entirely; just release the renderer.
            if self.renderer is not None:
                self.renderer.close()
                self.renderer = None
        if self._png_enabled:
            pylog.info(
                'CameraRecordingFrames wrote %d PNG frame(s) to %s',
                self.sample, self._frames_dir,
            )
