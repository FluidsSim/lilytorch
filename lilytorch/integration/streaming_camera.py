"""Streaming camera extensions – memory-efficient drop-ins for
``farms_mujoco.sensors.camera.CameraRecording`` and
``lilytorch.integration.camera_frame_recorder.CameraRecordingFrames``.

Problem
-------
The parent ``CameraRecording.__init__`` and ``initialize_episode`` both
pre-allocate a full ``[n_frames, H, W, 3]`` uint8 frame buffer.  At 4 K
(3840 × 2160) with ~600 frames this costs ≈ 14 GB *per camera instance*,
which causes OOM when two cameras are active simultaneously.

Solution
--------
When OpenCV is available (always the case in the lilytorch venv), each
rendered frame can be written directly to the ``cv2.VideoWriter`` and
reused in-place — no accumulation needed.  These subclasses keep only a
**single-row** scratch buffer ``self.data`` of shape ``[1, H, W, 3]`` and
render every frame into slot 0.  The rest of the rendering logic is
identical to the parent.

When cv2 is absent the classes transparently fall back to the parent's
full-buffer path so nothing breaks.

Usage
-----
Swap the ``loader`` key in any extension config::

    # fixed / top-down camera
    {
        "loader": "lilytorch.integration.streaming_camera.StreamingCameraRecording",
        "config": { ... same keys as CameraRecording ... },
    }

    # following camera with optional per-frame PNG export
    {
        "loader": "lilytorch.integration.streaming_camera.StreamingCameraRecordingFrames",
        "config": {
            ...same keys as CameraRecordingFrames...,
            "frames_dir"     : "...",   # optional
            "save_video"     : True,    # optional (default True)
            "png_compression": 3,       # optional (default 3)
        },
    }
"""

from __future__ import annotations

import os

import numpy as np
from farms_core import pylog
from farms_mujoco.sensors.camera import (
    CameraRecording,
    CameraRecordingOptions,
    USE_CV2,
)

if USE_CV2:
    import cv2


class StreamingCameraRecording(CameraRecording):
    """CameraRecording that keeps only one frame in RAM at a time.

    Each rendered frame is streamed directly to the ``cv2.VideoWriter``; the
    scratch buffer ``self.data`` is always a single row ``[1, H, W, 3]``.
    Falls back to the parent full-buffer behaviour when cv2 is absent.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace the huge buffer allocated by the parent __init__ immediately.
        if USE_CV2:
            self.data = np.zeros(
                [1, self.height, self.width, 3], dtype=np.uint8
            )
            self._streaming = True
        else:
            self._streaming = False

    def initialize_episode(self, task, physics):
        """Call the parent initialiser, then shrink the frame buffer to 1 row."""
        super().initialize_episode(task, physics)
        # The parent re-allocates self.data with the full buffer; shrink it.
        if USE_CV2:
            self.data = np.zeros(
                [1, self.height, self.width, 3], dtype=np.uint8
            )
            self._streaming = True

    def before_step(self, task, action, physics):
        """Render into slot 0 and stream to the cv2 writer; no accumulation."""
        if not self._streaming:
            # cv2 not available: use the parent's full-buffer path unchanged.
            super().before_step(task, action, physics)
            return

        del action
        if not task.iteration % (self.skips + 1):
            if self.viewer == 'dm_control':
                self.data[0, :, :, :] = physics.render(
                    width=self.width,
                    height=self.height,
                    camera_id=self.camera,
                )
            else:
                now = physics.time() / task.units.seconds
                timediff = now - self.last_capture
                self.last_capture = now
                self.camera.azimuth += self.angular_velocity * timediff
                self.camera.lookat[:] = np.array(self.offset)
                if self.links is not None:
                    self.camera.lookat[:] += np.array(
                        self.links.global_com_position(
                            iteration=task.iteration - 1,
                        )
                    )
                self.camera.distance = self.distance
                self.camera.elevation = self.elevation
                if self.renderer is not None:
                    self.renderer.update_scene(
                        physics.data.ptr,
                        camera=self.camera,
                        scene_option=self.render_options,
                    )
                    self.renderer.render(out=self.data[0, :, :, :])
            if self.out is not None:
                self.out.write(self.data[0, :, :, :][:, :, [2, 1, 0]])
            self.sample += 1


class StreamingCameraRecordingFrames(StreamingCameraRecording):
    """StreamingCameraRecording that also writes each frame as a PNG.

    Drop-in replacement for
    ``lilytorch.integration.camera_frame_recorder.CameraRecordingFrames``
    using the streaming (low-RAM) base class.
    """

    @classmethod
    def from_options(cls, config: dict, experiment_options):
        """From options – pops extra keys before passing to CameraRecordingOptions."""
        config = dict(config)
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
        """Set up the streaming recorder and create the frames directory."""
        super().initialize_episode(task, physics)

        if not self._png_enabled:
            pylog.warning(
                'StreamingCameraRecordingFrames: OpenCV (cv2) unavailable,'
                ' PNG export off'
            )

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
                'StreamingCameraRecordingFrames -> %s (save_video=%s)',
                self._frames_dir, self._save_video,
            )

    def before_step(self, task, action, physics):
        """Render via the streaming parent, then write the frame as a PNG."""
        prev_sample = self.sample
        super().before_step(task, action, physics)
        if not self._png_enabled:
            return
        if self.sample <= prev_sample:
            return  # no frame captured this iteration
        # self.data[0] always holds the latest RGB frame.
        bgr = cv2.cvtColor(self.data[0], cv2.COLOR_RGB2BGR)
        fname = os.path.join(
            self._frames_dir, f'frame_{self.sample - 1:06d}.png'
        )
        cv2.imwrite(
            fname, bgr, [cv2.IMWRITE_PNG_COMPRESSION, self._png_compression]
        )

    def end_episode(self, task, physics):
        """Finalise the video (if kept) and report the PNG count."""
        if self._save_video:
            super().end_episode(task, physics)
        else:
            if self.renderer is not None:
                self.renderer.close()
                self.renderer = None
        if self._png_enabled:
            pylog.info(
                'StreamingCameraRecordingFrames wrote %d PNG frame(s) to %s',
                self.sample, self._frames_dir,
            )
