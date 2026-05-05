"""GPU-backed 2-D flow viewer for MuJoCo.

This extension renders a 2-D scalar flow field through a GPU overlay in
an available MuJoCo renderer. The field processing stays on torch
tensors as long as possible, and each update uploads one RGB texture
directly from CUDA into an OpenGL texture owned by the MuJoCo renderer.

Compared with ``FlowViewer2D`` this removes the CPU-side hot path caused
by re-initialising thousands of box geoms every frame.

Usage
-----
Add to the simulation extensions list after ``FluidExtension``::

    {
        "loader": "lilytorch.integration.flow_viewer_2d_gpu.FlowViewer2DGPU",
        "config": {
            "field"                  : "curl",
            "nx_vis"                 : 96,
            "ny_vis"                 : 48,
            "alpha"                  : 0.55,
            "z_offset"               : 0.005,
            "vmin"                   : null,
            "vmax"                   : null,
            "smooth_sigma"           : 1.5,
            "crop_boundary"          : 2,
            "update_every"           : null,
        }
    }

Notes
-----
- When a ``mujoco.Renderer`` is present (for example via
    ``CameraRecording``), the extension uses a renderer-owned OpenGL quad
    overlay. That path does not depend on compiled MuJoCo textures or
    materials and does not round-trip through host memory.
- This extension no longer contains any CPU viewer fallback. When no
    GPU-capable renderer is present, it stays idle instead of updating the
    passive viewer. When the native MuJoCo GL hook is preloaded, the same
    CUDA texture is also drawn in the passive viewer without depending on
    ``CameraRecording``.
- MuJoCo textures are RGB-only, so the plane uses a uniform alpha.  The
  field still fades visually toward white near zero, but does not have
  per-pixel transparency like the CPU tile viewer.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import math
import os

import mujoco
import numpy as np
import torch
import torch.nn.functional as F

try:
        from OpenGL import GL
except ImportError:
        GL = None

from farms_core.experiment.options import ExperimentOptions
from farms_core.simulation.extensions import TaskExtension
from farms_mujoco.simulation.task import ExperimentTask
from dm_control.mjcf.physics import Physics


CUDA_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD = 2
CUDA_MEMCPY_DEVICE_TO_DEVICE = 3

_CUDART = None
_CUDART_INITIALIZED = False


def _load_cudart():
    global _CUDART, _CUDART_INITIALIZED

    if _CUDART_INITIALIZED:
        return _CUDART

    _CUDART_INITIALIZED = True
    if GL is None:
        return None

    lib_name = ctypes.util.find_library("cudart")
    if lib_name is None:
        return None

    lib = ctypes.CDLL(lib_name)
    lib.cudaGraphicsGLRegisterBuffer.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_uint,
        ctypes.c_uint,
    ]
    lib.cudaGraphicsGLRegisterBuffer.restype = ctypes.c_int
    lib.cudaGraphicsMapResources.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
    ]
    lib.cudaGraphicsMapResources.restype = ctypes.c_int
    lib.cudaGraphicsResourceGetMappedPointer.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
    ]
    lib.cudaGraphicsResourceGetMappedPointer.restype = ctypes.c_int
    lib.cudaGraphicsUnmapResources.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
    ]
    lib.cudaGraphicsUnmapResources.restype = ctypes.c_int
    lib.cudaGraphicsUnregisterResource.argtypes = [ctypes.c_void_p]
    lib.cudaGraphicsUnregisterResource.restype = ctypes.c_int
    lib.cudaMemcpy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    lib.cudaMemcpy.restype = ctypes.c_int
    lib.cudaGetErrorString.argtypes = [ctypes.c_int]
    lib.cudaGetErrorString.restype = ctypes.c_char_p

    _CUDART = lib
    return _CUDART


class _CudaGlTextureUploader:
    """Upload a CUDA tensor into a MuJoCo GL texture via a mapped PBO."""

    def __init__(self):
        self._cudart = _load_cudart()
        self._states: dict[int, dict] = {}

    @property
    def available(self) -> bool:
        return (
            self._cudart is not None
            and GL is not None
            and torch.cuda.is_available()
        )

    def release_renderer(self, renderer_id: int):
        state = self._states.pop(renderer_id, None)
        self._release_state(state)

    def close(self):
        for renderer_id in list(self._states):
            self.release_renderer(renderer_id)

    def upload(self, renderer, texture_slot: int, texture_u8: torch.Tensor) -> bool:
        if not self.available:
            raise RuntimeError(
                "CUDA-OpenGL texture upload is unavailable: need torch CUDA, "
                "PyOpenGL, and libcudart."
            )
        if not texture_u8.is_cuda:
            raise RuntimeError(
                "CUDA-OpenGL texture upload requires a CUDA tensor."
            )

        if texture_u8.dtype != torch.uint8 or texture_u8.ndim != 3 or texture_u8.shape[2] != 3:
            raise ValueError(
                "Expected a CUDA uint8 tensor with shape (height, width, 3)."
            )

        state = self._ensure_state(renderer, texture_slot, texture_u8.shape)

        renderer._gl_context.make_current()
        self._clear_gl_errors()
        torch.cuda.synchronize(texture_u8.device)

        resource = state["resource"]
        self._cuda_check(
            self._cudart.cudaGraphicsMapResources(1, ctypes.byref(resource), None),
            "cudaGraphicsMapResources",
        )
        try:
            ptr = ctypes.c_void_p()
            size = ctypes.c_size_t()
            self._cuda_check(
                self._cudart.cudaGraphicsResourceGetMappedPointer(
                    ctypes.byref(ptr),
                    ctypes.byref(size),
                    resource,
                ),
                "cudaGraphicsResourceGetMappedPointer",
            )
            if size.value < state["num_bytes"]:
                raise RuntimeError(
                    f"Mapped PBO too small: {size.value} < {state['num_bytes']}"
                )

            self._cuda_check(
                self._cudart.cudaMemcpy(
                    ptr,
                    ctypes.c_void_p(int(texture_u8.data_ptr())),
                    state["num_bytes"],
                    CUDA_MEMCPY_DEVICE_TO_DEVICE,
                ),
                "cudaMemcpy",
            )
        finally:
            self._cuda_check(
                self._cudart.cudaGraphicsUnmapResources(1, ctypes.byref(resource), None),
                "cudaGraphicsUnmapResources",
            )

        GL.glBindTexture(GL.GL_TEXTURE_2D, state["gl_texture"])
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, state["pbo"])
        GL.glTexSubImage2D(
            GL.GL_TEXTURE_2D,
            0,
            0,
            0,
            state["width"],
            state["height"],
            GL.GL_RGB,
            GL.GL_UNSIGNED_BYTE,
            ctypes.c_void_p(0),
        )
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        self._clear_gl_errors()
        return True

    def _ensure_state(self, renderer, texture_slot: int, shape: tuple[int, int, int]):
        renderer_id = id(renderer)
        height, width, channels = shape
        if channels != 3:
            raise ValueError("Expected RGB texture data.")

        renderer._gl_context.make_current()
        self._clear_gl_errors()
        gl_texture = int(renderer._mjr_context.texture[texture_slot])
        if GL.glIsTexture(gl_texture) != 1:
            mujoco.mjr_uploadTexture(renderer._model, renderer._mjr_context, texture_slot)
            self._clear_gl_errors()
            gl_texture = int(renderer._mjr_context.texture[texture_slot])
        if gl_texture <= 0 or GL.glIsTexture(gl_texture) != 1:
            raise RuntimeError(
                f"Renderer texture slot {texture_slot} did not expose a valid GL texture."
            )

        state = self._states.get(renderer_id)
        if (
            state is not None
            and state["width"] == width
            and state["height"] == height
            and state["texture_slot"] == texture_slot
            and state["gl_texture"] == gl_texture
        ):
            return state

        self.release_renderer(renderer_id)

        num_bytes = width * height * channels
        pbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, pbo)
        GL.glBufferData(GL.GL_PIXEL_UNPACK_BUFFER, num_bytes, None, GL.GL_STREAM_DRAW)
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)

        resource = ctypes.c_void_p()
        self._cuda_check(
            self._cudart.cudaGraphicsGLRegisterBuffer(
                ctypes.byref(resource),
                int(pbo),
                CUDA_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD,
            ),
            "cudaGraphicsGLRegisterBuffer",
        )

        state = {
            "renderer": renderer,
            "texture_slot": texture_slot,
            "gl_texture": gl_texture,
            "width": width,
            "height": height,
            "num_bytes": num_bytes,
            "pbo": pbo,
            "resource": resource,
        }
        self._states[renderer_id] = state
        return state

    def _release_state(self, state):
        if state is None:
            return

        resource = state.get("resource")
        if resource is not None and resource.value:
            try:
                self._cuda_check(
                    self._cudart.cudaGraphicsUnregisterResource(resource),
                    "cudaGraphicsUnregisterResource",
                )
            except Exception:
                pass

        pbo = state.get("pbo")
        if pbo and GL is not None:
            try:
                renderer = state.get("renderer")
                if renderer is not None and getattr(renderer, "_gl_context", None) is not None:
                    renderer._gl_context.make_current()
                GL.glDeleteBuffers(1, [int(pbo)])
            except Exception:
                pass

    @staticmethod
    def _cuda_check(err: int, call: str):
        if err != 0:
            message = None
            if _CUDART is not None:
                try:
                    message = _CUDART.cudaGetErrorString(err)
                except Exception:
                    message = None
            if message:
                raise RuntimeError(
                    f"{call} failed with CUDA error code {err}: {message.decode()}"
                )
            raise RuntimeError(f"{call} failed with CUDA error code {err}")

    @staticmethod
    def _clear_gl_errors():
        if GL is None:
            return
        while GL.glGetError() != GL.GL_NO_ERROR:
            pass


class _CudaGlQuadOverlay:
    """Upload a CUDA tensor into a renderer-owned GL texture and draw it."""

    _VERTEX_SHADER = """
    #version 330 core
    layout (location = 0) in vec3 aClip;
    layout (location = 1) in vec2 aUv;
    out vec2 vUv;
    void main() {
        vUv = aUv;
        gl_Position = vec4(aClip.xy, 0.0, aClip.z);
    }
    """

    _FRAGMENT_SHADER = """
    #version 330 core
    in vec2 vUv;
    out vec4 FragColor;
    uniform sampler2D uTexture;
    uniform float uAlpha;
    void main() {
        vec3 color = texture(uTexture, vUv).rgb;
        vec3 delta = vec3(1.0) - color;
        float strength = clamp(max(delta.r, max(delta.g, delta.b)), 0.0, 1.0);
        float alpha = uAlpha * (0.30 + 0.70 * pow(strength, 0.4));
        FragColor = vec4(color, alpha);
    }
    """

    def __init__(self):
        self._cudart = _load_cudart()
        self._states: dict[int, dict] = {}

    @property
    def available(self) -> bool:
        return (
            self._cudart is not None
            and GL is not None
            and torch.cuda.is_available()
        )

    def close(self):
        for renderer_id in list(self._states):
            self.release_renderer(renderer_id)

    def release_renderer(self, renderer_id: int):
        state = self._states.pop(renderer_id, None)
        self._release_state(state)

    def upload(self, renderer, texture_u8: torch.Tensor, synchronize: bool = True):
        if not self.available:
            raise RuntimeError(
                "CUDA-OpenGL overlay upload is unavailable: need torch CUDA, "
                "PyOpenGL, and libcudart."
            )
        if not texture_u8.is_cuda:
            raise RuntimeError("CUDA-OpenGL overlay upload requires a CUDA tensor.")
        if texture_u8.dtype != torch.uint8 or texture_u8.ndim != 3 or texture_u8.shape[2] != 3:
            raise ValueError(
                "Expected a CUDA uint8 tensor with shape (height, width, 3)."
            )

        state = self._ensure_state(renderer, texture_u8.shape)
        renderer._gl_context.make_current()
        self._clear_gl_errors()
        if synchronize:
            torch.cuda.current_stream(texture_u8.device).synchronize()

        resource = state["resource"]
        self._cuda_check(
            self._cudart.cudaGraphicsMapResources(1, ctypes.byref(resource), None),
            "cudaGraphicsMapResources",
        )
        try:
            ptr = ctypes.c_void_p()
            size = ctypes.c_size_t()
            self._cuda_check(
                self._cudart.cudaGraphicsResourceGetMappedPointer(
                    ctypes.byref(ptr),
                    ctypes.byref(size),
                    resource,
                ),
                "cudaGraphicsResourceGetMappedPointer",
            )
            if size.value < state["num_bytes"]:
                raise RuntimeError(
                    f"Mapped PBO too small: {size.value} < {state['num_bytes']}"
                )
            self._cuda_check(
                self._cudart.cudaMemcpy(
                    ptr,
                    ctypes.c_void_p(int(texture_u8.data_ptr())),
                    state["num_bytes"],
                    CUDA_MEMCPY_DEVICE_TO_DEVICE,
                ),
                "cudaMemcpy",
            )
        finally:
            self._cuda_check(
                self._cudart.cudaGraphicsUnmapResources(1, ctypes.byref(resource), None),
                "cudaGraphicsUnmapResources",
            )

        GL.glBindTexture(GL.GL_TEXTURE_2D, state["texture"])
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, state["pbo"])
        GL.glTexSubImage2D(
            GL.GL_TEXTURE_2D,
            0,
            0,
            0,
            state["width"],
            state["height"],
            GL.GL_RGB,
            GL.GL_UNSIGNED_BYTE,
            ctypes.c_void_p(0),
        )
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        self._clear_gl_errors()

    def draw(self, renderer, clip_vertices: np.ndarray, alpha: float):
        state = self._states.get(id(renderer))
        if state is None:
            raise RuntimeError("Renderer overlay state not initialized.")

        renderer._gl_context.make_current()
        vertices = np.ascontiguousarray(clip_vertices, dtype=np.float32)
        if vertices.shape != (4, 5):
            raise ValueError(
                f"Expected clip vertices with shape (4, 5), got {vertices.shape}."
            )

        self._clear_gl_errors()
        GL.glViewport(0, 0, renderer.width, renderer.height)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

        GL.glUseProgram(state["program"])
        GL.glUniform1f(state["alpha_location"], float(alpha))
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, state["texture"])
        GL.glUniform1i(state["texture_location"], 0)
        GL.glBindVertexArray(state["vao"])
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, state["vbo"])
        GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices)
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindVertexArray(0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glUseProgram(0)
        GL.glDisable(GL.GL_BLEND)
        GL.glEnable(GL.GL_DEPTH_TEST)
        self._clear_gl_errors()

    def _ensure_state(self, renderer, shape: tuple[int, int, int]):
        renderer_id = id(renderer)
        height, width, channels = shape
        if channels != 3:
            raise ValueError("Expected RGB texture data.")

        state = self._states.get(renderer_id)
        if state is not None and state["width"] == width and state["height"] == height:
            return state

        self.release_renderer(renderer_id)
        renderer._gl_context.make_current()
        self._clear_gl_errors()

        texture = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGB8,
            width,
            height,
            0,
            GL.GL_RGB,
            GL.GL_UNSIGNED_BYTE,
            None,
        )
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

        num_bytes = width * height * channels
        pbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, pbo)
        GL.glBufferData(GL.GL_PIXEL_UNPACK_BUFFER, num_bytes, None, GL.GL_STREAM_DRAW)
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)

        resource = ctypes.c_void_p()
        self._cuda_check(
            self._cudart.cudaGraphicsGLRegisterBuffer(
                ctypes.byref(resource),
                int(pbo),
                CUDA_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD,
            ),
            "cudaGraphicsGLRegisterBuffer",
        )

        program = self._create_program()
        vao = GL.glGenVertexArrays(1)
        vbo = GL.glGenBuffers(1)
        GL.glBindVertexArray(vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, 4 * 5 * 4, None, GL.GL_DYNAMIC_DRAW)
        stride = 5 * 4
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, False, stride, ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, False, stride, ctypes.c_void_p(12))
        GL.glEnableVertexAttribArray(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindVertexArray(0)

        state = {
            "renderer": renderer,
            "width": width,
            "height": height,
            "num_bytes": num_bytes,
            "texture": texture,
            "pbo": pbo,
            "resource": resource,
            "program": program,
            "vao": vao,
            "vbo": vbo,
            "alpha_location": GL.glGetUniformLocation(program, "uAlpha"),
            "texture_location": GL.glGetUniformLocation(program, "uTexture"),
        }
        self._states[renderer_id] = state
        self._clear_gl_errors()
        return state

    def _release_state(self, state):
        if state is None:
            return

        resource = state.get("resource")
        if resource is not None and resource.value:
            try:
                self._cuda_check(
                    self._cudart.cudaGraphicsUnregisterResource(resource),
                    "cudaGraphicsUnregisterResource",
                )
            except Exception:
                pass

        renderer = state.get("renderer")
        if renderer is not None and getattr(renderer, "_gl_context", None) is not None:
            try:
                renderer._gl_context.make_current()
            except Exception:
                renderer = None

        if GL is None:
            return
        try:
            if state.get("program"):
                GL.glDeleteProgram(int(state["program"]))
            if state.get("vbo"):
                GL.glDeleteBuffers(1, [int(state["vbo"])])
            if state.get("vao"):
                GL.glDeleteVertexArrays(1, [int(state["vao"])])
            if state.get("pbo"):
                GL.glDeleteBuffers(1, [int(state["pbo"])])
            if state.get("texture"):
                GL.glDeleteTextures([int(state["texture"])])
        except Exception:
            pass

    def _create_program(self):
        vertex = self._compile_shader(self._VERTEX_SHADER, GL.GL_VERTEX_SHADER)
        fragment = self._compile_shader(self._FRAGMENT_SHADER, GL.GL_FRAGMENT_SHADER)
        program = GL.glCreateProgram()
        GL.glAttachShader(program, vertex)
        GL.glAttachShader(program, fragment)
        GL.glLinkProgram(program)
        linked = GL.glGetProgramiv(program, GL.GL_LINK_STATUS)
        GL.glDeleteShader(vertex)
        GL.glDeleteShader(fragment)
        if not linked:
            raise RuntimeError(GL.glGetProgramInfoLog(program).decode())
        return program

    @staticmethod
    def _compile_shader(source: str, shader_type):
        shader = GL.glCreateShader(shader_type)
        GL.glShaderSource(shader, source)
        GL.glCompileShader(shader)
        compiled = GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS)
        if not compiled:
            raise RuntimeError(GL.glGetShaderInfoLog(shader).decode())
        return shader

    @staticmethod
    def _cuda_check(err: int, call: str):
        if err != 0:
            message = None
            if _CUDART is not None:
                try:
                    message = _CUDART.cudaGetErrorString(err)
                except Exception:
                    message = None
            if message:
                raise RuntimeError(
                    f"{call} failed with CUDA error code {err}: {message.decode()}"
                )
            raise RuntimeError(f"{call} failed with CUDA error code {err}")

    @staticmethod
    def _clear_gl_errors():
        if GL is None:
            return
        while GL.glGetError() != GL.GL_NO_ERROR:
            pass


def _field_curl(fs, u, v, p):
    del p
    return fs.vorticity(u, v)


def _field_pressure(fs, u, v, p):
    del fs, u, v
    return p


def _field_divergence(fs, u, v, p):
    del p
    return fs.divergence(u, v)


def _field_vel_mag(fs, u, v, p):
    del fs, p
    return torch.sqrt(u.square() + v.square())


FIELD_MAP_2D_GPU = {
    "curl": _field_curl,
    "vorticity": _field_curl,
    "pressure": _field_pressure,
    "divergence": _field_divergence,
    "vel_mag": _field_vel_mag,
}


def _torch_quantile_scalar(values: torch.Tensor, q: float) -> float:
    flat = values.reshape(-1)
    if flat.numel() == 0:
        return 1.0
    try:
        return float(torch.quantile(flat, q).detach().cpu())
    except RuntimeError:
        rank = max(1, min(flat.numel(), int(math.ceil(q * flat.numel()))))
        return float(flat.kthvalue(rank).values.detach().cpu())


def _gaussian_blur_2d(field: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return field

    radius = max(1, int(math.ceil(3.0 * sigma)))
    coords = torch.arange(
        -radius,
        radius + 1,
        device=field.device,
        dtype=field.dtype,
    )
    kernel = torch.exp(-0.5 * (coords / sigma).square())
    kernel = kernel / kernel.sum()

    data = field.unsqueeze(0).unsqueeze(0)
    kernel_x = kernel.view(1, 1, -1, 1)
    kernel_y = kernel.view(1, 1, 1, -1)

    data = F.pad(data, (0, 0, radius, radius), mode="replicate")
    data = F.conv2d(data, kernel_x)
    data = F.pad(data, (radius, radius, 0, 0), mode="replicate")
    data = F.conv2d(data, kernel_y)
    return data[0, 0]


def _resize_field(field: torch.Tensor, nx: int, ny: int) -> torch.Tensor:
    if field.shape == (nx, ny):
        return field
    return F.interpolate(
        field.unsqueeze(0).unsqueeze(0),
        size=(nx, ny),
        mode="bilinear",
        align_corners=False,
    )[0, 0]


def _rdbu_rgb_texture(field: torch.Tensor, scale: float, height: int, width: int) -> torch.Tensor:
    image_field = field.transpose(0, 1).contiguous()
    if image_field.shape != (height, width):
        image_field = F.interpolate(
            image_field.unsqueeze(0).unsqueeze(0),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

    t = torch.clamp(image_field / scale, -1.0, 1.0)  # (H, W), [-1, 1]
    # Match the legacy CPU viewer's linear RdBu-style RGB mapping.
    rgb = torch.stack([
        (1.0 + t.clamp(max=0.0)).clamp_(0.0, 1.0),
        (1.0 - t.abs()).clamp_(0.0, 1.0),
        (1.0 - t.clamp(min=0.0)).clamp_(0.0, 1.0),
    ], dim=-1)  # (H, W, 3) float32

    return (rgb * 255.0).round_().to(torch.uint8).contiguous()


class FlowViewer2DGPU(TaskExtension):
    """Render a 2-D flow field through a renderer-owned GPU overlay."""

    def __init__(
        self,
        experiment_options: ExperimentOptions,
        field: str = "curl",
        nx_vis: int = 96,
        ny_vis: int = 48,
        alpha: float = 0.55,
        z_offset: float = 0.005,
        vmin: float | None = None,
        vmax: float | None = None,
        smooth_sigma: float = 1.5,
        crop_boundary: int = 2,
        update_every: int | None = None,
        synchronize_cuda: bool = True,
    ):
        super().__init__()
        self.experiment_options = experiment_options
        self.field_name = field
        self.nx_vis = int(nx_vis)
        self.ny_vis = int(ny_vis)
        self.alpha = float(alpha)
        self.z_offset = float(z_offset)
        self.user_vmin = vmin
        self.user_vmax = vmax
        self.smooth_sigma = float(smooth_sigma)
        self.crop_boundary = int(crop_boundary)
        self.update_every = update_every
        self.synchronize_cuda = bool(synchronize_cuda)

        self._fluid_ext = None
        self._field_fn = FIELD_MAP_2D_GPU.get(field)
        self._iteration = 0
        self._initialized = False
        self._warned_3d = False

        self._render_resources_ready = False
        self._texture_width: int = 0
        self._texture_height: int = 0

        self._plane_center: np.ndarray | None = None
        self._plane_size: np.ndarray | None = None
        self._plane_center_f32: np.ndarray | None = None
        self._plane_size_xy_f32: np.ndarray | None = None

        self._renderer_patch_id: int | None = None

        self._texture_generation = 0
        self._last_renderer_upload_generation: dict[int, int] = {}
        self._renderer_texture_uploader = _CudaGlQuadOverlay()
        self._previous_texture_gpu: torch.Tensor | None = None
        self._latest_texture_gpu: torch.Tensor | None = None
        self._logged_renderer_gpu_notice = False
        self._logged_missing_renderer_notice = False
        self._gl_hook = None

    @classmethod
    def from_options(cls, config: dict, experiment_options: ExperimentOptions):
        return cls(
            experiment_options=experiment_options,
            field=config.get("field", "curl"),
            nx_vis=config.get("nx_vis", 96),
            ny_vis=config.get("ny_vis", 48),
            alpha=config.get("alpha", 0.55),
            z_offset=config.get("z_offset", 0.005),
            vmin=config.get("vmin", None),
            vmax=config.get("vmax", None),
            smooth_sigma=config.get("smooth_sigma", 1.5),
            crop_boundary=config.get("crop_boundary", 2),
            update_every=config.get("update_every", None),
            synchronize_cuda=config.get("synchronize_cuda", True),
        )

    def initialize_episode(self, task: ExperimentTask, physics: Physics):
        if self._initialized:
            return

        from lilytorch.integration.extensions import FluidExtension
        from lilytorch.integration.flow_viewer_gl_hook import get_flow_viewer_gl_hook

        for ext in task.extensions:
            if isinstance(ext, FluidExtension):
                self._fluid_ext = ext
                break

        if self._fluid_ext is None:
            print("[FlowViewer2DGPU] FluidExtension not found; disabled.")
            return

        if self._field_fn is None:
            print(
                f"[FlowViewer2DGPU] Unknown field '{self.field_name}'; "
                f"choose from {list(FIELD_MAP_2D_GPU)}. Disabled."
            )
            return

        self._gl_hook = get_flow_viewer_gl_hook()

        handler = getattr(self._fluid_ext, "BDIMhandler", None)
        fs = getattr(handler, "fluid_solver", None) if handler is not None else None
        if fs is not None:
            self._initialize_render_resources(task, fs)

        self._initialized = True

    def before_step(self, task: ExperimentTask, action, physics: Physics):
        del action
        del physics

        renderer = self._get_camera_renderer(task)
        gl_hook = self._gl_hook
        if renderer is None and gl_hook is None:
            if not self._logged_missing_renderer_notice:
                print(
                    "[FlowViewer2DGPU] No CameraRecording renderer or GL hook found; "
                    "GPU overlay disabled for this run."
                )
                self._logged_missing_renderer_notice = True
            return

        if self._fluid_ext is None or self._field_fn is None:
            self._iteration += 1
            return

        handler = getattr(self._fluid_ext, "BDIMhandler", None)
        if handler is None:
            return

        fs = getattr(handler, "fluid_solver", None)
        if fs is None:
            return

        if not self._render_resources_ready:
            self._initialize_render_resources(task, fs)
            if not self._render_resources_ready:
                return

        if renderer is not None:
            self._patch_camera_renderer(task)

        every = self.update_every or getattr(fs, "save_every", 200)
        iteration = getattr(handler, "iteration", self._iteration)
        self._iteration = iteration
        if iteration % every != 0:
            return

        u, v, p = fs.u0, fs.v0, fs.p0
        if getattr(fs, "w0", None) is not None:
            if not self._warned_3d:
                print(
                    "[FlowViewer2DGPU] 3D solver detected; "
                    "use FlowViewer for 3D fields. Skipping."
                )
                self._warned_3d = True
            return

        try:
            field = self._field_fn(fs, u, v, p)
        except Exception as exc:
            print(f"[FlowViewer2DGPU] field computation error: {exc}")
            return

        texture_rgb = self._build_texture_rgb(fs, field)
        if texture_rgb is None:
            return

        self._set_latest_texture(texture_rgb)
        if (
            gl_hook is not None
            and self._plane_center_f32 is not None
            and self._plane_size_xy_f32 is not None
        ):
            gl_hook.update(
                texture_rgb,
                self._plane_center_f32,
                self._plane_size_xy_f32,
                self.alpha,
                synchronize=self.synchronize_cuda,
            )

    def end_episode(self, task: ExperimentTask, physics: Physics):
        del task
        del physics
        self._renderer_texture_uploader.close()
        if self._gl_hook is not None:
            try:
                self._gl_hook.clear()
            except Exception:
                pass

    def _initialize_render_resources(self, task: ExperimentTask, fs):
        if self._render_resources_ready:
            return

        renderer = self._get_camera_renderer(task)
        gl_hook = self._gl_hook
        if renderer is not None and not self._renderer_texture_uploader.available:
            raise RuntimeError(
                "[FlowViewer2DGPU] CUDA-OpenGL overlay upload is unavailable: "
                "need torch CUDA, PyOpenGL, and libcudart."
            )
        if renderer is None and gl_hook is None:
            return

        self._texture_width = max(2, self.nx_vis)
        self._texture_height = max(2, self.ny_vis)

        xmin, xmax = float(fs.xmin), float(fs.xmax)
        ymin, ymax = float(fs.ymin), float(fs.ymax)
        self._plane_center = np.array(
            [(xmin + xmax) * 0.5, (ymin + ymax) * 0.5, self.z_offset],
            dtype=np.float64,
        )
        self._plane_size = np.array(
            [(xmax - xmin) * 0.5, (ymax - ymin) * 0.5, 5e-4],
            dtype=np.float64,
        )
        self._plane_center_f32 = self._plane_center.astype(np.float32, copy=True)
        self._plane_size_xy_f32 = self._plane_size[:2].astype(np.float32, copy=True)

        if renderer is not None:
            neutral = torch.full(
                (self._texture_height, self._texture_width, 3),
                255,
                dtype=torch.uint8,
                device="cuda",
            )
            self._set_latest_texture(neutral)
            self._patch_camera_renderer(task)

        self._render_resources_ready = True
        print(
            "[FlowViewer2DGPU] Ready"
            + (f" — renderer GL overlay ({self._texture_width}x{self._texture_height})" if renderer is not None else "")
            + (" + GL hook (passive viewer)" if gl_hook is not None else "")
            + "."
        )

    def _build_texture_rgb(self, fs, field) -> torch.Tensor | None:
        if not torch.is_tensor(field):
            field = torch.as_tensor(field)
        if field.ndim != 2:
            print(
                f"[FlowViewer2DGPU] Expected a 2D field, got shape {tuple(field.shape)}."
            )
            return None

        field = field.detach().to(dtype=torch.float32)

        c = self.crop_boundary
        if c > 0 and min(field.shape) > 2 * c:
            field = field[c:-c, c:-c]

        if self.smooth_sigma > 0:
            field = _gaussian_blur_2d(field, self.smooth_sigma)

        if hasattr(fs, "composite_body") and hasattr(fs.composite_body, "sdf_val"):
            sdf = fs.composite_body.sdf_val
            if torch.is_tensor(sdf):
                sdf = sdf.detach().to(device=field.device)
                if c > 0 and min(sdf.shape) > 2 * c:
                    sdf = sdf[c:-c, c:-c]
                if sdf.shape == field.shape:
                    field = field.masked_fill(sdf < 0, 0.0)

        vmin = self.user_vmin
        vmax = self.user_vmax
        if vmin is None or vmax is None:
            abs_field = field.abs()
            limit = _torch_quantile_scalar(abs_field, 0.99)
            if limit < 1e-12:
                limit = float(abs_field.max().detach().cpu()) or 1.0
            if vmin is None:
                vmin = -limit
            if vmax is None:
                vmax = limit

        field_vis = _resize_field(field, self.nx_vis, self.ny_vis)
        scale = max(abs(vmin), abs(vmax))
        if scale < 1e-15:
            scale = 1.0

        return _rdbu_rgb_texture(
            field_vis,
            scale=scale,
            height=self._texture_height,
            width=self._texture_width,
        )

    def _set_latest_texture(self, texture_rgb: torch.Tensor):
        texture_rgb = texture_rgb.detach().contiguous()
        if not texture_rgb.is_cuda:
            raise RuntimeError(
                "[FlowViewer2DGPU] FlowViewer2DGPU requires CUDA textures; "
                "CPU field updates are not supported."
            )
        self._texture_generation += 1
        self._previous_texture_gpu = self._latest_texture_gpu
        self._latest_texture_gpu = texture_rgb

    def _upload_texture_to_renderer(self, renderer):
        if not self._render_resources_ready:
            return
        renderer_id = id(renderer)
        if self._last_renderer_upload_generation.get(renderer_id) == self._texture_generation:
            return

        if self._latest_texture_gpu is None:
            raise RuntimeError(
                "FlowViewer2DGPU expected a CUDA texture for renderer upload, "
                "but the latest texture is not on CUDA."
            )

        self._renderer_texture_uploader.upload(
            renderer,
            self._latest_texture_gpu,
            synchronize=self.synchronize_cuda,
        )
        if not self._logged_renderer_gpu_notice:
            print(
                "[FlowViewer2DGPU] Using CUDA-OpenGL quad overlay uploads for "
                "CameraRecording renderer."
            )
            self._logged_renderer_gpu_notice = True

        self._last_renderer_upload_generation[renderer_id] = self._texture_generation

    def _patch_camera_renderer(self, task: ExperimentTask):
        renderer = self._get_camera_renderer(task)
        if renderer is None:
            return
        if self._renderer_patch_id == id(renderer):
            return

        original_render = renderer.render
        flow_viewer = self

        def _render_with_gpu_plane(out=None):
            if renderer._depth_rendering or renderer._segmentation_rendering:
                return original_render(out=out)

            if renderer._mjr_context is None:
                raise RuntimeError("render cannot be called after close.")

            if renderer._gl_context:
                renderer._gl_context.make_current()

            out_shape = (renderer.height, renderer.width, 3)
            if out is None:
                result = np.empty(out_shape, dtype=np.uint8)
            else:
                if out.shape != out_shape:
                    raise ValueError(
                        f"Expected `out.shape == {out_shape}`. Got `out.shape={out.shape}` instead."
                    )
                result = out

            mujoco._render.mjr_render(renderer._rect, renderer._scene, renderer._mjr_context)

            flow_viewer._upload_texture_to_renderer(renderer)
            flow_viewer._draw_renderer_overlay(renderer)
            mujoco._render.mjr_readPixels(result, None, renderer._rect, renderer._mjr_context)
            if renderer._gl_context:
                result[:] = np.flipud(result)
            return result

        renderer.render = _render_with_gpu_plane
        self._renderer_patch_id = id(renderer)
        print("[FlowViewer2DGPU] Patched CameraRecording renderer.")

    @staticmethod
    def _get_camera_renderer(task: ExperimentTask):
        for ext in task.extensions:
            if type(ext).__name__ == "CameraRecording":
                return getattr(ext, "renderer", None)
        return None

    def _draw_renderer_overlay(self, renderer):
        clip_vertices = self._compute_overlay_clip_vertices(renderer)
        self._renderer_texture_uploader.draw(renderer, clip_vertices, self.alpha)

    def _compute_overlay_clip_vertices(self, renderer) -> np.ndarray:
        if self._plane_center is None or self._plane_size is None:
            raise RuntimeError("FlowViewer2DGPU overlay geometry is not initialized.")

        headpos = np.zeros(3, dtype=np.float64)
        forward = np.zeros(3, dtype=np.float64)
        up = np.zeros(3, dtype=np.float64)
        mujoco.mjv_cameraInModel(headpos, forward, up, renderer.scene)

        forward_norm = np.linalg.norm(forward)
        up_norm = np.linalg.norm(up)
        if forward_norm < 1e-8 or up_norm < 1e-8:
            raise RuntimeError("FlowViewer2DGPU could not resolve the renderer camera basis.")
        forward /= forward_norm
        up /= up_norm

        right = np.cross(forward, up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-8:
            raise RuntimeError("FlowViewer2DGPU could not resolve the renderer right axis.")
        right /= right_norm

        frustum_height = float(mujoco.mjv_frustumHeight(renderer.scene))
        if frustum_height <= 0:
            raise RuntimeError("FlowViewer2DGPU received a non-positive renderer frustum height.")

        half_height_unit = 0.5 * frustum_height
        half_width_unit = half_height_unit * (float(renderer.width) / float(renderer.height))
        z = float(self._plane_center[2])
        x0 = float(self._plane_center[0] - self._plane_size[0])
        x1 = float(self._plane_center[0] + self._plane_size[0])
        y0 = float(self._plane_center[1] - self._plane_size[1])
        y1 = float(self._plane_center[1] + self._plane_size[1])

        corners = (
            ((x0, y0, z), (0.0, 0.0)),
            ((x1, y0, z), (1.0, 0.0)),
            ((x0, y1, z), (0.0, 1.0)),
            ((x1, y1, z), (1.0, 1.0)),
        )
        vertices = np.empty((4, 5), dtype=np.float32)

        for index, (point_xyz, uv) in enumerate(corners):
            rel = np.array(point_xyz, dtype=np.float64) - headpos
            depth = float(np.dot(rel, forward))
            if depth <= 1e-6:
                raise RuntimeError(
                    "FlowViewer2DGPU overlay plane is behind the CameraRecording camera."
                )
            x_cam = float(np.dot(rel, right))
            y_cam = float(np.dot(rel, up))
            vertices[index, 0] = x_cam / half_width_unit
            vertices[index, 1] = y_cam / half_height_unit
            vertices[index, 2] = depth
            vertices[index, 3] = float(uv[0])
            vertices[index, 4] = float(uv[1])

        return vertices


# Backward-compatible export so configs can swap the module path without
# having to change the class name.
FlowViewer2D = FlowViewer2DGPU


_FLOW_VIEWER_2D_GPU_LOADERS = {
    "lilytorch.integration.flow_viewer_2d_gpu.FlowViewer2D",
    "lilytorch.integration.flow_viewer_2d_gpu.FlowViewer2DGPU",
}


def prepare_flow_viewer_2d_gpu_env(
    base_env: dict | None = None,
    simulation_extensions: list[dict] | None = None,
) -> dict:
    """Prepare LD_PRELOAD only for runs that actually use FlowViewer2DGPU."""
    env = dict(os.environ if base_env is None else base_env)
    extensions = [] if simulation_extensions is None else simulation_extensions

    if not any(
        isinstance(ext, dict) and ext.get("loader") in _FLOW_VIEWER_2D_GPU_LOADERS
        for ext in extensions
    ):
        return env

    from lilytorch.integration.flow_viewer_gl_hook import prepare_mujoco_gl_hook_env

    return prepare_mujoco_gl_hook_env(env)