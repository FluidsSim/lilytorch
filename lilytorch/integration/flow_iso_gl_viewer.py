"""
FlowIsoGLViewer – render a true triangulated 3D isosurface inside the
MuJoCo passive viewer via an ``LD_PRELOAD``-injected OpenGL hook.

Architecture
------------
1. An ``mjr_render`` hook is compiled from embedded C source and
   ``LD_PRELOAD``-ed into the FARMS subprocess (same dynamic-linker
   technique as ``flow_viewer_gl_hook.py`` – the two libraries can
   coexist in ``LD_PRELOAD`` and chain through ``dlsym(RTLD_NEXT)``).
2. After MuJoCo finishes its own render pass, the hook draws a
   triangulated isosurface mesh whose vertex buffer is shared with a
   CUDA tensor via ``cudaGraphicsGLRegisterBuffer`` (zero-copy upload).
3. Each frame the ``FlowIsoGLViewer`` extension:
   - extracts the chosen field from the GPU solver,
   - optionally smooths it with a 3D Gaussian on GPU (separable convs),
   - runs marching cubes (GPU via ``torchmcubes`` if installed, else
     CPU via ``skimage``),
   - builds an interleaved (pos, normal, color) float32 tensor on GPU,
   - calls ``hook.update(...)`` which atomically swaps the live mesh.

The drawn mesh participates in MuJoCo's depth buffer, so the textured
animat correctly occludes/is-occluded by the isosurface (unlike the
primitive-tile fallback in ``flow_iso_viewer.py``).

Usage
-----
1. Wrap the FARMS subprocess launch::

       from lilytorch.integration.flow_iso_gl_viewer import prepare_iso_gl_hook_env
       subprocess.run(['bash', 'run.sh'], env=prepare_iso_gl_hook_env())

2. Add the extension to the sim config **after** ``FluidExtension``::

       {
           "loader": "lilytorch.integration.flow_iso_gl_viewer.FlowIsoGLViewer",
           "config": {
               "field":         "omega_z",
               "iso_fraction":  0.15,
               "iso_value":     null,
               "smooth_sigma":  2.5,
               "crop_boundary": 3,
               "alpha":         0.55,
               "update_every":  1,
               "max_vertices":  300000,
               "mc_backend":    "auto",
               "exclude_body":  true,
           }
       }

Notes
-----
- Recorded video (``CameraRecording``) uses a separate offscreen GL
  context and is NOT covered by this hook. Use the tile-based
  ``FlowIsoViewer`` for that case if needed.
- Requires ``gcc`` and the MuJoCo headers bundled with the ``mujoco`` Python
  package. (CUDA is no longer required for the GL hook; mesh data is sent
  via a CPU memcpy so the same hook works in both the viewer's GL context
  and the offscreen renderer's context.)
- Optional but recommended: ``pip install torchmcubes`` for GPU
  marching cubes (gives realtime update on 256^3 grids).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np
import torch


# ── Embedded C source ─────────────────────────────────────────────────────────

_GL_HOOK_C_SOURCE = r"""
#define _GNU_SOURCE
#define GL_GLEXT_PROTOTYPES

#include <GL/gl.h>
#include <GL/glext.h>
#include <dlfcn.h>
#include <math.h>
#include <mujoco/mujoco.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef void (*mjr_render_fn)(mjrRect viewport, mjvScene* scn, const mjrContext* con);

#define ISO_FLOATS_PER_VERTEX 9
#define ISO_MIN_CAPACITY 4096
#define ISO_MAX_CTX 4

/* Per-GL-context cache. Each mjrContext* is tied to a specific GL context;
   resources (VBO/VAO/program) only exist in the GL context where they were
   created, so we keep separate ones for the interactive window and the
   offscreen CameraRecording renderer. */
typedef struct {
  const mjrContext* key;   /* NULL = unused slot */
  GLuint vbo;
  GLuint vao;
  GLuint program;
  GLint  loc_cam_pos;
  GLint  loc_cam_right;
  GLint  loc_cam_up;
  GLint  loc_cam_fwd;
  GLint  loc_half_w;
  GLint  loc_half_h;
  GLint  loc_light_dir;
  GLint  loc_alpha;
  int    vbo_capacity;
  int    uploaded_generation;
} CtxResources;

typedef struct {
  pthread_mutex_t mutex;
  int enabled;
  int debug_force_visible;

  /* CPU-side mirror of the interleaved mesh data (pos3, normal3, rgb3). */
  unsigned char* cpu_mesh;
  size_t cpu_mesh_capacity;
  int vertex_count;
  int generation;

  /* Render-side parameters. */
  float alpha;
  float znear;
  float zfar;
  float light_dir[3];

  CtxResources ctx[ISO_MAX_CTX];
} FlowIsoHookState;

static FlowIsoHookState g_state = {
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .enabled = 0,
    .debug_force_visible = 0,
    .cpu_mesh = NULL,
    .cpu_mesh_capacity = 0,
    .vertex_count = 0,
    .generation = 0,
    .alpha = 0.55f,
    .znear = 0.02f,
    .zfar = 100.0f,
    .light_dir = {0.3f, 0.4f, -1.0f},
};

static mjr_render_fn real_mjr_render = NULL;
static int logged_gl_error = 0;

/* Optional overlay text drawn on top of every mjr_render call.
   Updated by Python via lily_flow_iso_hook_set_overlay_text. */
#define ISO_OVERLAY_TEXT_MAX 256
static char g_overlay_text[ISO_OVERLAY_TEXT_MAX] = "";
static int  g_overlay_grid = 0;  /* mjtGridPos: TOPLEFT=0, TOPRIGHT=1, BOTTOMLEFT=2, BOTTOMRIGHT=3 */

static const char* k_vertex_shader =
    "#version 330 core\n"
    "layout (location = 0) in vec3 aPos;\n"
    "layout (location = 1) in vec3 aNormal;\n"
    "layout (location = 2) in vec3 aColor;\n"
    "uniform vec3  uCamPos;\n"
    "uniform vec3  uCamRight;\n"
    "uniform vec3  uCamUp;\n"
    "uniform vec3  uCamFwd;\n"
    "uniform float uHalfW;\n"
    "uniform float uHalfH;\n"
    "out vec3 vNormalWorld;\n"
    "out vec3 vColor;\n"
    "void main() {\n"
    "  vec3 rel = aPos - uCamPos;\n"
    "  float depth = dot(rel, uCamFwd);\n"
    "  float xeye  = dot(rel, uCamRight);\n"
    "  float yeye  = dot(rel, uCamUp);\n"
    "  /* No depth test will be used; emit z=0 so fragments aren't clipped. */\n"
    "  gl_Position = vec4(xeye / uHalfW, yeye / uHalfH, 0.0, depth);\n"
    "  vNormalWorld = aNormal;\n"
    "  vColor = aColor;\n"
    "}\n";

static const char* k_fragment_shader =
    "#version 330 core\n"
    "in vec3 vNormalWorld;\n"
    "in vec3 vColor;\n"
    "out vec4 FragColor;\n"
    "uniform vec3  uLightDir;\n"
    "uniform float uAlpha;\n"
    "void main() {\n"
    "  vec3 N = normalize(vNormalWorld);\n"
    "  vec3 L = normalize(-uLightDir);\n"
    "  float diff = max(abs(dot(N, L)), 0.0);\n"
    "  float shade = 0.25 + 0.75 * diff;\n"
    "  FragColor = vec4(vColor * shade, uAlpha);\n"
    "}\n";

static void ensure_real_mjr_render(void) {
  if (!real_mjr_render) {
    real_mjr_render = (mjr_render_fn)dlsym(RTLD_NEXT, "mjr_render");
  }
}

static int compile_shader(GLenum type, const char* src, GLuint* out) {
  GLuint sh = glCreateShader(type);
  GLint ok = 0;
  glShaderSource(sh, 1, &src, NULL);
  glCompileShader(sh);
  glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
  if (!ok) {
    if (!logged_gl_error) {
      char log[4096]; GLsizei n = 0;
      glGetShaderInfoLog(sh, sizeof(log), &n, log);
      fprintf(stderr, "[FlowIsoGLHook] shader compile failed: %.*s\n", (int)n, log);
      logged_gl_error = 1;
    }
    glDeleteShader(sh);
    return 0;
  }
  *out = sh;
  return 1;
}

static int create_program_in_ctx(CtxResources* c) {
  GLuint vs = 0, fs = 0, prog = 0;
  GLint linked = 0;
  if (!compile_shader(GL_VERTEX_SHADER, k_vertex_shader, &vs)) return 0;
  if (!compile_shader(GL_FRAGMENT_SHADER, k_fragment_shader, &fs)) {
    glDeleteShader(vs);
    return 0;
  }
  prog = glCreateProgram();
  glAttachShader(prog, vs);
  glAttachShader(prog, fs);
  glLinkProgram(prog);
  glDeleteShader(vs);
  glDeleteShader(fs);
  glGetProgramiv(prog, GL_LINK_STATUS, &linked);
  if (!linked) {
    if (!logged_gl_error) {
      char log[4096]; GLsizei n = 0;
      glGetProgramInfoLog(prog, sizeof(log), &n, log);
      fprintf(stderr, "[FlowIsoGLHook] link failed: %.*s\n", (int)n, log);
      logged_gl_error = 1;
    }
    glDeleteProgram(prog);
    return 0;
  }
  c->program       = prog;
  c->loc_cam_pos   = glGetUniformLocation(prog, "uCamPos");
  c->loc_cam_right = glGetUniformLocation(prog, "uCamRight");
  c->loc_cam_up    = glGetUniformLocation(prog, "uCamUp");
  c->loc_cam_fwd   = glGetUniformLocation(prog, "uCamFwd");
  c->loc_half_w    = glGetUniformLocation(prog, "uHalfW");
  c->loc_half_h    = glGetUniformLocation(prog, "uHalfH");
  c->loc_light_dir = glGetUniformLocation(prog, "uLightDir");
  c->loc_alpha     = glGetUniformLocation(prog, "uAlpha");
  return 1;
}

static int next_capacity(int needed) {
  int cap = ISO_MIN_CAPACITY;
  while (cap < needed) cap *= 2;
  return cap;
}

static CtxResources* find_or_make_ctx_locked(const mjrContext* key) {
  /* Existing entry. */
  for (int i = 0; i < ISO_MAX_CTX; ++i) {
    if (g_state.ctx[i].key == key) return &g_state.ctx[i];
  }
  /* First free slot. */
  for (int i = 0; i < ISO_MAX_CTX; ++i) {
    if (g_state.ctx[i].key == NULL) {
      memset(&g_state.ctx[i], 0, sizeof(CtxResources));
      g_state.ctx[i].key = key;
      g_state.ctx[i].uploaded_generation = -1;
      return &g_state.ctx[i];
    }
  }
  return NULL;  /* cache full */
}

static int ensure_ctx_resources(CtxResources* c) {
  /* Program is context-local but capacity-independent — keep across resizes. */
  if (!c->program) {
    if (!create_program_in_ctx(c)) return 0;
  }
  if (c->vbo && c->vao && c->vbo_capacity >= g_state.vertex_count) {
    return 1;
  }
  /* Need (re)allocation. Tear down existing buffer/array (keep program). */
  if (c->vbo) { glDeleteBuffers(1, &c->vbo); c->vbo = 0; }
  if (c->vao) { glDeleteVertexArrays(1, &c->vao); c->vao = 0; }

  int cap = next_capacity(g_state.vertex_count > 0 ? g_state.vertex_count : ISO_MIN_CAPACITY);

  glGenBuffers(1, &c->vbo);
  glBindBuffer(GL_ARRAY_BUFFER, c->vbo);
  glBufferData(GL_ARRAY_BUFFER,
               (GLsizeiptr)cap * ISO_FLOATS_PER_VERTEX * (GLsizeiptr)sizeof(float),
               NULL, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  glGenVertexArrays(1, &c->vao);
  glBindVertexArray(c->vao);
  glBindBuffer(GL_ARRAY_BUFFER, c->vbo);
  const GLsizei stride = (GLsizei)(ISO_FLOATS_PER_VERTEX * sizeof(float));
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (const void*)0);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, (const void*)(3 * sizeof(float)));
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, (const void*)(6 * sizeof(float)));
  glEnableVertexAttribArray(2);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);

  c->vbo_capacity = cap;
  c->uploaded_generation = -1;
  return 1;
}

static int upload_ctx(CtxResources* c) {
  if (g_state.vertex_count <= 0) return 0;
  if (!g_state.cpu_mesh) return 0;
  if (c->uploaded_generation == g_state.generation) return 1;
  size_t bytes = (size_t)g_state.vertex_count * ISO_FLOATS_PER_VERTEX * sizeof(float);
  glBindBuffer(GL_ARRAY_BUFFER, c->vbo);
  glBufferSubData(GL_ARRAY_BUFFER, 0, (GLsizeiptr)bytes, g_state.cpu_mesh);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  c->uploaded_generation = g_state.generation;
  return 1;
}

static int draw_iso_call_count = 0;
static void drain_gl_errors_silent(void) {
  /* Empty the error queue without logging. Used on hook entry where the
     pending error is leftover state from MuJoCo's own draw — not ours. */
  int max = 8;
  while (glGetError() != GL_NO_ERROR && max-- > 0) {}
}
static void drain_gl_errors_verbose(const char* tag) {
  GLenum e;
  int max = 6;
  while ((e = glGetError()) != GL_NO_ERROR && max-- > 0) {
    fprintf(stderr, "[FlowIsoGLHook] GL error 0x%x at %s\n", e, tag);
  }
}
static void draw_iso_locked(CtxResources* c, const mjvScene* scn, mjrRect viewport) {
  mjtNum headpos[3], forward[3], up[3], right[3];

  /* Clear any pending error from MuJoCo's own draw so our diagnostics
     reflect only errors from our own calls. Silent — these aren't ours. */
  drain_gl_errors_silent();

  mjv_cameraInModel(headpos, forward, up, scn);
  double fn = sqrt(forward[0]*forward[0] + forward[1]*forward[1] + forward[2]*forward[2]);
  double un = sqrt(up[0]*up[0] + up[1]*up[1] + up[2]*up[2]);
  if (fn < 1e-8 || un < 1e-8) return;
  for (int i = 0; i < 3; ++i) { forward[i] /= fn; up[i] /= un; }
  right[0] = forward[1]*up[2] - forward[2]*up[1];
  right[1] = forward[2]*up[0] - forward[0]*up[2];
  right[2] = forward[0]*up[1] - forward[1]*up[0];
  double rn = sqrt(right[0]*right[0] + right[1]*right[1] + right[2]*right[2]);
  if (rn < 1e-8) return;
  for (int i = 0; i < 3; ++i) right[i] /= rn;

  double frust_h = mjv_frustumHeight(scn);
  if (frust_h <= 0.0) return;
  double half_h = 0.5 * frust_h;
  double half_w = half_h * ((double)viewport.width / (double)viewport.height);

  if (draw_iso_call_count < 3) {
    fprintf(stderr,
            "[FlowIsoGLHook] draw call %d: ctx=%p verts=%d viewport=(%d,%d %dx%d) "
            "prog=%u vao=%u vbo=%u cap=%d\n",
            draw_iso_call_count, (const void*)c->key, g_state.vertex_count,
            viewport.left, viewport.bottom, viewport.width, viewport.height,
            c->program, c->vao, c->vbo, c->vbo_capacity);
    draw_iso_call_count++;
  }

  if (!upload_ctx(c)) return;

  /* Save state. */
  GLboolean depth_test = glIsEnabled(GL_DEPTH_TEST);
  GLboolean blend      = glIsEnabled(GL_BLEND);
  GLboolean cull_face  = glIsEnabled(GL_CULL_FACE);
  GLint prev_prog = 0;
  glGetIntegerv(GL_CURRENT_PROGRAM, &prev_prog);

  glViewport(viewport.left, viewport.bottom, viewport.width, viewport.height);
  /* The iso is a translucent overlay; we don't try to depth-test against
     MuJoCo's body geometry because the HUD pass invalidates the projection
     matrix stack and we lack a reliable way to match MuJoCo's depth values.
     debug_force_visible kept for compatibility — additionally turns alpha off. */
  glDisable(GL_DEPTH_TEST);
  if (g_state.debug_force_visible) {
    glDisable(GL_BLEND);          /* opaque */
  } else {
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  }
  glDisable(GL_CULL_FACE);

  glUseProgram(c->program);
  glUniform3f(c->loc_cam_pos,   (float)headpos[0], (float)headpos[1], (float)headpos[2]);
  glUniform3f(c->loc_cam_right, (float)right[0],   (float)right[1],   (float)right[2]);
  glUniform3f(c->loc_cam_up,    (float)up[0],      (float)up[1],      (float)up[2]);
  glUniform3f(c->loc_cam_fwd,   (float)forward[0], (float)forward[1], (float)forward[2]);
  glUniform1f(c->loc_half_w,    (float)half_w);
  glUniform1f(c->loc_half_h,    (float)half_h);
  glUniform3f(c->loc_light_dir, g_state.light_dir[0], g_state.light_dir[1], g_state.light_dir[2]);
  glUniform1f(c->loc_alpha,     g_state.alpha);

  glBindVertexArray(c->vao);
  glDrawArrays(GL_TRIANGLES, 0, g_state.vertex_count);
  glBindVertexArray(0);

  if (draw_iso_call_count <= 3) {
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
      fprintf(stderr, "[FlowIsoGLHook] glDrawArrays error: 0x%x\n", err);
    }
  }

  /* Restore. */
  glUseProgram((GLuint)prev_prog);
  if (!blend) glDisable(GL_BLEND);
  if (cull_face) glEnable(GL_CULL_FACE);
  if (depth_test) glEnable(GL_DEPTH_TEST);
}

__attribute__((visibility("default")))
void lily_flow_iso_hook_set_overlay_text(const char* text, int grid_pos) {
  pthread_mutex_lock(&g_state.mutex);
  if (text) {
    strncpy(g_overlay_text, text, ISO_OVERLAY_TEXT_MAX - 1);
    g_overlay_text[ISO_OVERLAY_TEXT_MAX - 1] = '\0';
  } else {
    g_overlay_text[0] = '\0';
  }
  if (grid_pos >= 0 && grid_pos <= 3) g_overlay_grid = grid_pos;
  pthread_mutex_unlock(&g_state.mutex);
}

__attribute__((visibility("default")))
void lily_flow_iso_hook_set_enabled(int enabled) {
  pthread_mutex_lock(&g_state.mutex);
  g_state.enabled = enabled ? 1 : 0;
  pthread_mutex_unlock(&g_state.mutex);
}

__attribute__((visibility("default")))
void lily_flow_iso_hook_clear(void) {
  pthread_mutex_lock(&g_state.mutex);
  g_state.enabled = 0;
  g_state.vertex_count = 0;
  pthread_mutex_unlock(&g_state.mutex);
}

__attribute__((visibility("default")))
void lily_flow_iso_hook_set_render_params(
    float alpha,
    float znear,
    float zfar,
    const float light_dir[3]) {
  pthread_mutex_lock(&g_state.mutex);
  g_state.alpha = alpha;
  if (znear > 0.0f) g_state.znear = znear;
  if (zfar  > znear) g_state.zfar  = zfar;
  if (light_dir) memcpy(g_state.light_dir, light_dir, 3 * sizeof(float));
  pthread_mutex_unlock(&g_state.mutex);
}

__attribute__((visibility("default")))
void lily_flow_iso_hook_set_debug_force_visible(int enabled) {
  pthread_mutex_lock(&g_state.mutex);
  g_state.debug_force_visible = enabled ? 1 : 0;
  pthread_mutex_unlock(&g_state.mutex);
}

__attribute__((visibility("default")))
void lily_flow_iso_hook_update(
    const void* cpu_ptr,
    int vertex_count,
    int generation) {
  pthread_mutex_lock(&g_state.mutex);
  g_state.enabled = 1;
  g_state.vertex_count = vertex_count;
  g_state.generation = generation;

  size_t bytes = (size_t)vertex_count * ISO_FLOATS_PER_VERTEX * sizeof(float);
  if (g_state.cpu_mesh_capacity < bytes) {
    free(g_state.cpu_mesh);
    g_state.cpu_mesh = (unsigned char*)malloc(bytes ? bytes : 1);
    g_state.cpu_mesh_capacity = g_state.cpu_mesh ? bytes : 0;
  }
  if (cpu_ptr && g_state.cpu_mesh && bytes > 0) {
    memcpy(g_state.cpu_mesh, cpu_ptr, bytes);
  }
  pthread_mutex_unlock(&g_state.mutex);
}

__attribute__((visibility("default")))
void mjr_render(mjrRect viewport, mjvScene* scn, const mjrContext* con) {
  ensure_real_mjr_render();
  if (real_mjr_render) real_mjr_render(viewport, scn, con);
  if (!con || !scn) return;

  /* Draw the iso in WHICHEVER GL context this mjr_render belongs to.
     Both the window framebuffer (interactive viewer) and offscreen FBOs
     (CameraRecording's mujoco.Renderer) get our iso composited on top. */
  pthread_mutex_lock(&g_state.mutex);
  if (g_state.enabled && g_state.vertex_count > 0) {
    CtxResources* c = find_or_make_ctx_locked(con);
    if (c && ensure_ctx_resources(c)) {
      draw_iso_locked(c, scn, viewport);
    }
  }

  /* Optional text overlay (e.g. simulation time). Drawn after iso, so it
     appears on top of both MuJoCo's scene and our iso surface. */
  if (g_overlay_text[0]) {
    char buf[ISO_OVERLAY_TEXT_MAX];
    int grid = g_overlay_grid;
    memcpy(buf, g_overlay_text, ISO_OVERLAY_TEXT_MAX);
    pthread_mutex_unlock(&g_state.mutex);
    mjr_overlay(mjFONT_NORMAL, grid, viewport, buf, NULL, con);
  } else {
    pthread_mutex_unlock(&g_state.mutex);
  }
}
"""


# ── Build helpers (largely mirrored from flow_viewer_gl_hook.py) ─────────────

_HOOK_ENV_VAR = "LILYTORCH_FLOW_ISO_GL_HOOK_LIB"
_BUILD_LOCK = threading.Lock()
_HOOK_INSTANCE: "FlowIsoGLHook | None" = None


def _locate_cudart() -> Path | None:
    candidates: list[Path] = []
    found = ctypes.util.find_library("cudart")
    if found:
        p = Path(found)
        if p.is_absolute() and p.exists():
            candidates.append(p)
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        home = Path(cuda_home)
        candidates += [
            home / "targets" / "x86_64-linux" / "lib" / "libcudart.so",
            home / "lib64" / "libcudart.so",
        ]
    for root in sorted(Path("/usr/local").glob("cuda*")):
        candidates += [
            root / "targets" / "x86_64-linux" / "lib" / "libcudart.so",
            root / "lib64" / "libcudart.so",
        ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return None


def _find_libmujoco(pkg: Path) -> Path | None:
    for c in sorted(pkg.glob("libmujoco.so*")):
        if c.is_file():
            return c
    return None


def ensure_iso_gl_hook_built() -> str:
    """Compile the embedded C source into a shared library and return its path."""
    with _BUILD_LOCK:
        gcc = shutil.which("gcc")
        if gcc is None:
            raise RuntimeError("gcc is required to build the iso-surface GL hook.")

        import mujoco
        pkg = Path(mujoco.__file__).resolve().parent
        inc = pkg / "include"
        libmj = _find_libmujoco(pkg)
        if libmj is None:
            raise RuntimeError(f"Could not find libmujoco.so in {pkg}.")

        build_dir = Path(__file__).resolve().parent.parent.parent / "build" / "flow_iso_gl_hook"
        out = build_dir / "libflow_iso_gl_hook.so"
        src_mtime = Path(__file__).stat().st_mtime

        if out.exists() and out.stat().st_mtime >= src_mtime:
            return str(out)

        build_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            suffix=".c", delete=False, mode="w", dir=build_dir,
        ) as tmp:
            tmp.write(_GL_HOOK_C_SOURCE)
            tmp_c = tmp.name
        try:
            cmd = [
                gcc, "-shared", "-fPIC", "-O3", "-std=c11",
                "-Wall", "-Wextra", "-Wno-unused-parameter",
                "-I", str(inc),
                f"-Wl,-rpath,{pkg}",
                "-o", str(out),
                tmp_c,
                str(libmj),
                "-ldl", "-lGL", "-lpthread", "-lm",
            ]
            r = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if r.returncode != 0:
                raise RuntimeError(
                    "Failed to build iso GL hook:\nSTDOUT:\n"
                    f"{r.stdout}\nSTDERR:\n{r.stderr}"
                )
        finally:
            try:
                os.unlink(tmp_c)
            except OSError:
                pass
        return str(out)


class FlowIsoGLHook:
    """Python controller for the iso-surface GL hook library."""

    def __init__(self, library_path: str):
        self.library_path = library_path
        self._generation = 0
        self._lib = ctypes.CDLL(library_path, mode=ctypes.RTLD_GLOBAL)

        self._lib.lily_flow_iso_hook_set_enabled.argtypes = [ctypes.c_int]
        self._lib.lily_flow_iso_hook_set_enabled.restype = None
        self._lib.lily_flow_iso_hook_clear.argtypes = []
        self._lib.lily_flow_iso_hook_clear.restype = None
        self._lib.lily_flow_iso_hook_set_render_params.argtypes = [
            ctypes.c_float, ctypes.c_float, ctypes.c_float,
            ctypes.POINTER(ctypes.c_float),
        ]
        self._lib.lily_flow_iso_hook_set_render_params.restype = None
        self._lib.lily_flow_iso_hook_set_debug_force_visible.argtypes = [ctypes.c_int]
        self._lib.lily_flow_iso_hook_set_debug_force_visible.restype = None
        self._lib.lily_flow_iso_hook_set_overlay_text.argtypes = [
            ctypes.c_char_p, ctypes.c_int,
        ]
        self._lib.lily_flow_iso_hook_set_overlay_text.restype = None
        self._lib.lily_flow_iso_hook_update.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ]
        self._lib.lily_flow_iso_hook_update.restype = None

    def set_enabled(self, enabled: bool):
        self._lib.lily_flow_iso_hook_set_enabled(1 if enabled else 0)

    def set_debug_force_visible(self, enabled: bool):
        self._lib.lily_flow_iso_hook_set_debug_force_visible(1 if enabled else 0)

    def set_overlay_text(self, text: str, grid_pos: int = 0):
        """Set on-screen text drawn over every frame (viewer + recorded video).

        ``grid_pos``: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right.
        Pass an empty string to clear.
        """
        self._lib.lily_flow_iso_hook_set_overlay_text(
            (text or "").encode("utf-8"), ctypes.c_int(int(grid_pos)),
        )

    def clear(self):
        self._lib.lily_flow_iso_hook_clear()

    def set_render_params(
        self,
        alpha: float,
        znear: float,
        zfar: float,
        light_dir: tuple[float, float, float] = (0.3, 0.4, -1.0),
    ):
        arr = (ctypes.c_float * 3)(*[float(c) for c in light_dir])
        self._lib.lily_flow_iso_hook_set_render_params(
            ctypes.c_float(float(alpha)),
            ctypes.c_float(float(znear)),
            ctypes.c_float(float(zfar)),
            arr,
        )

    def update(self, interleaved_xyz_nrm_rgb: torch.Tensor, synchronize: bool = True):
        """Submit a new mesh; (V, 9) float32 tensor (pos, normal, color).

        Accepts either CUDA or CPU tensor. The C side keeps a private mirror
        of the bytes (memcpy from this contiguous buffer), so per-GL-context
        VBO uploads work in both the live viewer and the offscreen renderer.
        """
        t = interleaved_xyz_nrm_rgb
        if t.dtype != torch.float32 or t.ndim != 2 or t.shape[1] != 9:
            raise ValueError(
                f"Expected (V, 9) float32 tensor, got shape {tuple(t.shape)} "
                f"dtype {t.dtype}."
            )
        if t.is_cuda:
            if synchronize:
                torch.cuda.current_stream(t.device).synchronize()
            t_cpu = t.detach().to('cpu', non_blocking=False).contiguous()
        else:
            t_cpu = t.detach().contiguous()
        self._generation += 1
        # Keep the CPU tensor alive long enough for the C side to memcpy it.
        self._last_cpu_mesh = t_cpu
        self._lib.lily_flow_iso_hook_update(
            ctypes.c_void_p(int(t_cpu.data_ptr())),
            ctypes.c_int(int(t_cpu.shape[0])),
            ctypes.c_int(self._generation),
        )


_ISO_VIEWER_LOADERS = (
    "lilytorch.integration.flow_iso_gl_viewer.FlowIsoGLViewer",
)


def prepare_iso_gl_hook_env(
    base_env: dict | None = None,
    simulation_extensions: list[dict] | None = None,
) -> dict:
    """Return an env dict with LD_PRELOAD set to the iso-surface GL hook library.

    If ``simulation_extensions`` is provided and none of them reference
    ``FlowIsoGLViewer``, this function returns the env unchanged (so it can
    be safely chained for runs that do not need the iso hook).

    Compose with ``prepare_mujoco_gl_hook_env`` / ``prepare_flow_viewer_2d_gpu_env``:
    each library inserts a single LD_PRELOAD entry; chaining via
    ``dlsym(RTLD_NEXT)`` keeps both ``mjr_render`` interceptions composable.
    """
    env = dict(os.environ if base_env is None else base_env)

    if simulation_extensions is not None and not any(
        isinstance(ext, dict) and ext.get("loader") in _ISO_VIEWER_LOADERS
        for ext in simulation_extensions
    ):
        return env

    if (
        sys.platform.startswith("linux")
        and env.get("DISPLAY")
        and env.get("MUJOCO_GL", "glfw") == "glfw"
        and env.get("XDG_SESSION_TYPE", "x11") == "x11"
        and shutil.which("nvidia-smi") is not None
    ):
        env.setdefault("__NV_PRIME_RENDER_OFFLOAD", "1")
        env.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")
        env.setdefault("__VK_LAYER_NV_optimus", "NVIDIA_only")

    try:
        lib = ensure_iso_gl_hook_built()
    except Exception as e:
        print(f"[FlowIsoGLHook] WARNING: could not build GL hook: {e}")
        return env

    entries = [p for p in env.get("LD_PRELOAD", "").split(":") if p]
    if lib not in entries:
        entries.insert(0, lib)
    env["LD_PRELOAD"] = ":".join(entries)
    env[_HOOK_ENV_VAR] = lib
    return env


def get_iso_gl_hook() -> "FlowIsoGLHook | None":
    global _HOOK_INSTANCE
    if _HOOK_INSTANCE is not None:
        return _HOOK_INSTANCE
    lib = os.environ.get(_HOOK_ENV_VAR)
    if not lib or not os.path.exists(lib):
        return None
    _HOOK_INSTANCE = FlowIsoGLHook(lib)
    return _HOOK_INSTANCE


# ── Field extractors (kept local so the extension is self-contained) ────────

def _vort(fs, u, v, w):
    return fs.vorticity_components(u, v, w)

def _field_omega_x(fs, u, v, p, w): return _vort(fs, u, v, w)["omega_x"]
def _field_omega_y(fs, u, v, p, w): return _vort(fs, u, v, w)["omega_y"]
def _field_omega_z(fs, u, v, p, w): return _vort(fs, u, v, w)["omega_z"]
def _field_omega_mag(fs, u, v, p, w): return _vort(fs, u, v, w)["omega_mag"]
def _field_vel_mag(fs, u, v, p, w): return (u**2 + v**2 + w**2).sqrt()
def _field_pressure(fs, u, v, p, w): return p

FIELD_MAP = {
    "omega_x":   _field_omega_x,
    "omega_y":   _field_omega_y,
    "omega_z":   _field_omega_z,
    "omega_mag": _field_omega_mag,
    "vel_mag":   _field_vel_mag,
    "pressure":  _field_pressure,
}


# ── GPU helpers ──────────────────────────────────────────────────────────────

def _separable_gaussian_3d(field: torch.Tensor, sigma: float) -> torch.Tensor:
    """In-place-friendly separable 3D Gaussian smoothing on a CUDA tensor.

    ``field`` is (Nx, Ny, Nz). Returns smoothed tensor of same shape.
    """
    if sigma <= 0:
        return field
    radius = max(1, int(round(3.0 * sigma)))
    xs = torch.arange(-radius, radius + 1, device=field.device, dtype=field.dtype)
    k = torch.exp(-(xs * xs) / (2.0 * sigma * sigma))
    k = (k / k.sum()).view(1, 1, -1)
    pad = (radius, radius)
    f = field.unsqueeze(0).unsqueeze(0)  # (1, 1, Nx, Ny, Nz)
    # axis x
    f1 = torch.nn.functional.conv1d(
        f.reshape(-1, 1, field.shape[0]),
        k, padding=pad[0],
    ).reshape(1, 1, field.shape[0], field.shape[1], field.shape[2])
    # axis y
    f2 = torch.nn.functional.conv1d(
        f1.permute(0, 1, 3, 2, 4).reshape(-1, 1, field.shape[1]),
        k, padding=pad[0],
    ).reshape(1, 1, field.shape[1], field.shape[0], field.shape[2]).permute(0, 1, 3, 2, 4)
    # axis z
    f3 = torch.nn.functional.conv1d(
        f2.permute(0, 1, 4, 3, 2).reshape(-1, 1, field.shape[2]),
        k, padding=pad[0],
    ).reshape(1, 1, field.shape[2], field.shape[1], field.shape[0]).permute(0, 1, 4, 3, 2)
    return f3.squeeze(0).squeeze(0).contiguous()


def _normals_from_field_gradient(
    field: torch.Tensor,
    verts_idx: torch.Tensor,  # (V, 3) integer indices into field
    sign: float,              # +1 for positive iso (outward = grad), -1 for negative iso
) -> torch.Tensor:
    """Per-vertex normals via central-difference gradient sampled at nearest cell.

    For an iso-level c of a scalar f, the outward normal of the surface (pointing
    away from the f > c side) is ``-sign * grad f / |grad f|``.
    """
    Nx, Ny, Nz = field.shape
    ix = verts_idx[:, 0].clamp(1, Nx - 2)
    iy = verts_idx[:, 1].clamp(1, Ny - 2)
    iz = verts_idx[:, 2].clamp(1, Nz - 2)
    gx = 0.5 * (field[ix + 1, iy, iz] - field[ix - 1, iy, iz])
    gy = 0.5 * (field[ix, iy + 1, iz] - field[ix, iy - 1, iz])
    gz = 0.5 * (field[ix, iy, iz + 1] - field[ix, iy, iz - 1])
    n = torch.stack([gx, gy, gz], dim=1)
    norm = n.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return -sign * (n / norm)


def _bases_from_normals_np(normals: np.ndarray) -> np.ndarray:
    """Build (M, 3, 3) rotation matrices whose 3rd column is each normal.

    Used to orient flat-ellipsoid "tile" geoms in the offscreen renderer.
    """
    n = np.asarray(normals, dtype=np.float64)
    mag = np.linalg.norm(n, axis=1)
    bad = (~np.isfinite(mag)) | (mag < 1e-12)
    n = np.where(bad[:, None], np.array([[0.0, 0.0, 1.0]]), n)
    mag = np.where(bad, 1.0, mag)
    n = n / mag[:, None]
    use_z = (np.abs(n[:, 2]) < 0.9)
    helper = np.where(use_z[:, None],
                      np.array([[0.0, 0.0, 1.0]]),
                      np.array([[1.0, 0.0, 0.0]]))
    t1 = np.cross(helper, n)
    t1 = t1 / np.maximum(np.linalg.norm(t1, axis=1, keepdims=True), 1e-12)
    t2 = np.cross(n, t1)
    return np.stack([t1, t2, n], axis=2)  # (M, 3, 3)


# ── The extension ────────────────────────────────────────────────────────────

try:
    from farms_core.simulation.extensions import TaskExtension as _BaseExtension
except ImportError:  # pragma: no cover – fallback for environments without farms
    class _BaseExtension:  # type: ignore[no-redef]
        def __init__(self):
            pass


class FlowIsoGLViewer(_BaseExtension):
    """Realtime isosurface in the MuJoCo viewer via the GL hook."""

    def __init__(
        self,
        experiment_options,
        field: str = "omega_z",
        iso_fraction: float = 0.15,
        iso_value: float | None = None,
        smooth_sigma: float = 2.5,
        crop_boundary: int = 3,
        alpha: float = 0.55,
        update_every: int = 1,
        max_vertices: int = 300_000,
        mc_backend: str = "auto",     # "auto" | "torchmcubes" | "skimage"
        exclude_body: bool = True,
        light_dir: tuple[float, float, float] = (0.3, 0.4, -1.0),
        diag_every: int = 200,
        debug_force_visible: bool = False,
        video_tiles: bool = False,
        video_max_tiles: int = 4000,
        video_tile_size: float = 0.001,
        video_tile_thickness: float = 0.0003,
        video_tile_alpha: float = 0.7,
        show_time: bool = True,
        time_format: str = "t = {t:.3f} s",
        time_grid_pos: int = 0,
    ):
        super().__init__()
        self.experiment_options = experiment_options
        self.field_name = field
        self.iso_fraction = float(iso_fraction)
        self.iso_value = float(iso_value) if iso_value is not None else None
        self.smooth_sigma = float(smooth_sigma)
        self.crop_boundary = int(crop_boundary)
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.update_every = max(1, int(update_every))
        self.max_vertices = int(max_vertices)
        self.mc_backend = str(mc_backend)
        self.exclude_body = bool(exclude_body)
        self.light_dir = tuple(float(c) for c in light_dir)
        self.diag_every = max(1, int(diag_every))
        self.debug_force_visible = bool(debug_force_visible)
        self.video_tiles = bool(video_tiles)
        self.video_max_tiles = max(1, int(video_max_tiles))
        self.video_tile_size = float(video_tile_size)
        self.video_tile_thickness = float(video_tile_thickness)
        self.video_tile_alpha = float(np.clip(video_tile_alpha, 0.0, 1.0))

        # State for the offscreen-renderer tile path (CameraRecording).
        self._video_camera_renderer_patched = False
        self._video_positions: np.ndarray | None = None
        self._video_mats: np.ndarray | None = None
        self._video_colors: np.ndarray | None = None
        self._video_n_active = 0
        self._video_tile_size_arr = np.array(
            [self.video_tile_size, self.video_tile_size, self.video_tile_thickness],
            dtype=np.float64,
        )
        self.show_time = bool(show_time)
        self.time_format = str(time_format)
        self.time_grid_pos = int(time_grid_pos)

        self._fluid_ext = None
        self._field_fn = FIELD_MAP.get(field)
        self._iteration = 0
        self._initialized = False
        self._hook: FlowIsoGLHook | None = None
        self._mc_callable = None
        self._mc_name = "unset"
        self._warned_2d = False
        self._warned_no_hook = False
        # Persistent device buffer for the interleaved mesh (avoids
        # reallocation per frame; resized only when overflowing).
        self._mesh_buffer: torch.Tensor | None = None

    # FARMS factory
    @classmethod
    def from_options(cls, config: dict, experiment_options):
        return cls(
            experiment_options=experiment_options,
            field=config.get("field", "omega_z"),
            iso_fraction=config.get("iso_fraction", 0.15),
            iso_value=config.get("iso_value", None),
            smooth_sigma=config.get("smooth_sigma", 2.5),
            crop_boundary=config.get("crop_boundary", 3),
            alpha=config.get("alpha", 0.55),
            update_every=config.get("update_every", 1),
            max_vertices=config.get("max_vertices", 300_000),
            mc_backend=config.get("mc_backend", "auto"),
            exclude_body=config.get("exclude_body", True),
            light_dir=tuple(config.get("light_dir", (0.3, 0.4, -1.0))),
            diag_every=config.get("diag_every", 200),
            debug_force_visible=config.get("debug_force_visible", False),
            video_tiles=config.get("video_tiles", False),
            video_max_tiles=config.get("video_max_tiles", 4000),
            video_tile_size=config.get("video_tile_size", 0.001),
            video_tile_thickness=config.get("video_tile_thickness", 0.0003),
            video_tile_alpha=config.get("video_tile_alpha", 0.7),
            show_time=config.get("show_time", True),
            time_format=config.get("time_format", "t = {t:.3f} s"),
            time_grid_pos=config.get("time_grid_pos", 0),
        )

    # ── lifecycle ────────────────────────────────────────────────────────
    def initialize_episode(self, task, physics):
        if self._initialized:
            return

        from lilytorch.integration.extensions import FluidExtension
        for ext in task.extensions:
            if isinstance(ext, FluidExtension):
                self._fluid_ext = ext
                break
        if self._fluid_ext is None:
            print("[FlowIsoGLViewer] FluidExtension not found – disabled.")
            return

        if self._field_fn is None:
            print(f"[FlowIsoGLViewer] Unknown field '{self.field_name}', "
                  f"choose from: {list(FIELD_MAP)}. Disabled.")
            return

        self._hook = get_iso_gl_hook()
        if self._hook is None:
            print(
                "[FlowIsoGLViewer] WARNING: GL hook library not preloaded. "
                "Wrap your subprocess launch with "
                "`prepare_iso_gl_hook_env()` (see flow_iso_gl_viewer.py docs). "
                "Extension disabled."
            )
            return

        self._mc_callable, self._mc_name = self._select_mc_backend()
        if self._mc_callable is None:
            print(
                "[FlowIsoGLViewer] No marching cubes backend available. "
                "Install scikit-image (CPU) or torchmcubes (GPU). Disabled."
            )
            return

        # Read znear/zfar from MuJoCo model so the iso depth lines up with
        # the textured bodies.
        try:
            m = physics.model.ptr  # dm_control wraps the raw mjModel here
            extent = float(m.stat.extent)
            znear = float(m.vis.map.znear) * extent
            zfar  = float(m.vis.map.zfar) * extent
        except Exception:
            znear, zfar = 0.02, 100.0
        self._hook.set_render_params(self.alpha, znear, zfar, self.light_dir)
        self._hook.set_debug_force_visible(self.debug_force_visible)
        self._hook.set_enabled(True)

        print(
            f"[FlowIsoGLViewer] MC backend: {self._mc_name}; "
            f"max_vertices={self.max_vertices}; alpha={self.alpha:.2f}; "
            f"znear={znear:.4g}, zfar={zfar:.4g}; "
            f"video_tiles={self.video_tiles}."
        )

        if self.video_tiles:
            self._video_positions = np.zeros((self.video_max_tiles, 3), dtype=np.float64)
            self._video_mats      = np.zeros((self.video_max_tiles, 9), dtype=np.float64)
            self._video_colors    = np.zeros((self.video_max_tiles, 4), dtype=np.float32)
            self._patch_camera_renderer(task)

        self._initialized = True

    def _patch_camera_renderer(self, task):
        """Monkey-patch CameraRecording's offscreen renderer so the iso surface
        appears in the recorded video. Same pattern as flow_iso_viewer.py:
        before each render call, inject oriented flat ellipsoid geoms built
        from our current MC mesh into ``renderer.scene``.
        """
        if self._video_camera_renderer_patched:
            return

        cam_ext = None
        for ext in task.extensions:
            if type(ext).__name__ == 'CameraRecording':
                cam_ext = ext
                break
        if cam_ext is None:
            if not getattr(self, '_warned_no_cam_ext', False):
                print("[FlowIsoGLViewer] No CameraRecording extension found – "
                      "video tile injection disabled.")
                self._warned_no_cam_ext = True
            return
        renderer = getattr(cam_ext, 'renderer', None)
        if renderer is None:
            # Renderer not yet built; before_step will retry next call.
            return

        import mujoco as _mj
        self_ref = self
        original_render = renderer.render
        _eye3 = np.eye(3, dtype=np.float64).ravel()

        def _render_with_tiles(out=None):
            n = self_ref._video_n_active
            if n > 0 and self_ref._video_positions is not None:
                scn = renderer.scene
                for i in range(n):
                    if scn.ngeom >= scn.maxgeom:
                        break
                    g = scn.geoms[scn.ngeom]
                    _mj.mjv_initGeom(
                        g,
                        _mj.mjtGeom.mjGEOM_ELLIPSOID,
                        self_ref._video_tile_size_arr,
                        self_ref._video_positions[i],
                        self_ref._video_mats[i],
                        self_ref._video_colors[i],
                    )
                    scn.ngeom += 1
            return original_render(out=out)

        renderer.render = _render_with_tiles
        self._video_camera_renderer_patched = True
        print("[FlowIsoGLViewer] Patched CameraRecording renderer for video output.")

    def _update_video_tiles(self, mesh: torch.Tensor):
        """Subsample the (V, 9) mesh tensor to up to video_max_tiles vertices
        and stage them as oriented flat ellipsoids in the persistent numpy
        buffers consumed by the patched offscreen renderer.
        """
        if not self.video_tiles or self._video_positions is None:
            return
        V = int(mesh.shape[0])
        if V == 0:
            self._video_n_active = 0
            return

        n_keep = min(V, self.video_max_tiles)
        if V > n_keep:
            idx = torch.randperm(V, device=mesh.device)[:n_keep]
            sel = mesh.index_select(0, idx)
        else:
            sel = mesh
        sel_cpu = sel.detach().to('cpu', torch.float32).numpy()

        pos = sel_cpu[:, 0:3].astype(np.float64, copy=False)
        nrm = sel_cpu[:, 3:6].astype(np.float64, copy=False)
        rgb = sel_cpu[:, 6:9]

        bases = _bases_from_normals_np(nrm)  # (n_keep, 3, 3)
        mats = bases.reshape(n_keep, 9)

        self._video_positions[:n_keep] = pos
        self._video_mats[:n_keep] = mats
        self._video_colors[:n_keep, :3] = rgb
        self._video_colors[:n_keep, 3] = self.video_tile_alpha
        self._video_n_active = n_keep

    def _select_mc_backend(self):
        want = self.mc_backend.lower()
        if want in ("auto", "torchmcubes"):
            try:
                from torchmcubes import marching_cubes as _tmc  # type: ignore
                return (_tmc, "torchmcubes")
            except ImportError:
                if want == "torchmcubes":
                    return (None, "none")
        try:
            from skimage.measure import marching_cubes as _smc
            return (_smc, "skimage")
        except ImportError:
            return (None, "none")

    def before_step(self, task, action, physics):
        if not self._initialized:
            self._iteration += 1
            return

        # Deferred renderer patch: CameraRecording's renderer is often created
        # lazily, after initialize_episode runs. Retry until it shows up.
        if self.video_tiles and not self._video_camera_renderer_patched:
            self._patch_camera_renderer(task)

        # Update the on-screen time overlay (shown in both viewer + video).
        if self.show_time:
            try:
                t_sim = float(physics.data.time)
                self._hook.set_overlay_text(
                    self.time_format.format(t=t_sim), self.time_grid_pos,
                )
            except Exception:
                pass

        handler = getattr(self._fluid_ext, "BDIMhandler", None)
        if handler is None:
            return
        fs = getattr(handler, "fluid_solver", None)
        if fs is None:
            return

        iteration = getattr(handler, "iteration", self._iteration)
        self._iteration = iteration
        if iteration % self.update_every != 0:
            return

        u, v, w, p = fs.u0, fs.v0, getattr(fs, "w0", None), fs.p0
        if w is None:
            if not self._warned_2d:
                print("[FlowIsoGLViewer] WARNING: 2D solver – requires 3D. Skipping.")
                self._warned_2d = True
            return

        try:
            mesh = self._build_mesh(fs, u, v, p, w)
        except Exception as e:
            print(f"[FlowIsoGLViewer] mesh build error: {e}")
            mesh = None

        # Periodic diagnostics: report what the extension is producing.
        if iteration % max(1, self.diag_every) == 0:
            self._print_diag(fs, u, v, p, w, mesh)

        if mesh is None or mesh.shape[0] == 0:
            self._hook.clear()
            self._hook.set_enabled(True)
            self._video_n_active = 0
            return

        self._hook.update(mesh, synchronize=True)
        if self.video_tiles:
            self._update_video_tiles(mesh)

    def _print_diag(self, fs, u, v, p, w, mesh):
        try:
            field = self._field_fn(fs, u, v, p, w)
            if not isinstance(field, torch.Tensor):
                field = torch.as_tensor(field)
            field = field.detach()
            fmin = float(field.min().item())
            fmax = float(field.max().item())
            peak = max(abs(fmin), abs(fmax))
            thresh = (
                self.iso_value if self.iso_value is not None
                else self.iso_fraction * peak
            )
            n_vertices = int(mesh.shape[0]) if mesh is not None else 0
            extra = ""
            if mesh is not None and n_vertices > 0:
                pos = mesh[:, :3]
                pmin = pos.min(dim=0).values.tolist()
                pmax = pos.max(dim=0).values.tolist()
                extra = (
                    f" pos_min=({pmin[0]:.4f},{pmin[1]:.4f},{pmin[2]:.4f})"
                    f" pos_max=({pmax[0]:.4f},{pmax[1]:.4f},{pmax[2]:.4f})"
                )
            print(
                f"[FlowIsoGLViewer] iter={self._iteration} "
                f"field={self.field_name} range=[{fmin:.3e},{fmax:.3e}] "
                f"peak={peak:.3e} thresh={thresh:.3e} mc={self._mc_name} "
                f"vertices={n_vertices}{extra}"
            )
        except Exception as e:
            print(f"[FlowIsoGLViewer] diag error: {e}")

    # ── mesh extraction ──────────────────────────────────────────────────
    def _build_mesh(self, fs, u, v, p, w):
        """Run MC + per-vertex normals; return (V, 9) float32 CUDA tensor."""
        field = self._field_fn(fs, u, v, p, w)  # torch.Tensor on GPU
        if not isinstance(field, torch.Tensor):
            field = torch.as_tensor(field, device=u.device)
        field = field.detach().to(torch.float32)

        c = self.crop_boundary
        if c > 0:
            field = field[c:-c, c:-c, c:-c].contiguous()

        if self.smooth_sigma > 0:
            field = _separable_gaussian_3d(field, self.smooth_sigma)

        # SDF mask (move to same shape as field)
        sdf = None
        if self.exclude_body and hasattr(fs, "composite_body") \
                and hasattr(fs.composite_body, "sdf_val"):
            sdf = fs.composite_body.sdf_val.detach().to(field.device, torch.float32)
            if c > 0:
                sdf = sdf[c:-c, c:-c, c:-c]

        # Threshold
        if sdf is not None:
            abs_f = field.abs()[sdf > 0]
        else:
            abs_f = field.abs()
        peak = float(abs_f.max().item()) if abs_f.numel() > 0 else 0.0

        if self.iso_value is not None and self.iso_value > 0:
            threshold = float(self.iso_value)
        elif peak < 1e-12:
            return None
        else:
            threshold = self.iso_fraction * peak

        fmin = float(field.min().item())
        fmax = float(field.max().item())
        bipolar = (fmin < -1e-12) and (fmax > 1e-12)

        # World coords for vertex placement
        x = fs.x.detach().to(field.device, torch.float32)
        y = fs.y.detach().to(field.device, torch.float32)
        z = fs.z.detach().to(field.device, torch.float32)
        if c > 0:
            x, y, z = x[c:-c], y[c:-c], z[c:-c]
        dx = float((x[-1] - x[0]).item()) / max(1, x.numel() - 1)
        dy = float((y[-1] - y[0]).item()) / max(1, y.numel() - 1)
        dz = float((z[-1] - z[0]).item()) / max(1, z.numel() - 1)
        origin = torch.tensor(
            [float(x[0].item()), float(y[0].item()), float(z[0].item())],
            device=field.device, dtype=torch.float32,
        )

        chunks: list[torch.Tensor] = []

        def _run(level: float, color_rgb: tuple[float, float, float], sign: float):
            vt = self._run_mc(field, level, (dx, dy, dz), origin, sdf)
            if vt is None:
                return
            tri_idx, tri_pos = vt
            if tri_pos.shape[0] == 0:
                return
            # Per-vertex normals via field gradient at nearest cell.
            n = _normals_from_field_gradient(field, tri_idx, sign)
            color = torch.tensor(color_rgb, device=field.device, dtype=torch.float32)
            color = color.expand(tri_pos.shape[0], 3)
            chunks.append(torch.cat([tri_pos, n, color], dim=1))

        if bipolar:
            _run(threshold,  (0.95, 0.20, 0.18), sign=+1.0)
            _run(-threshold, (0.18, 0.40, 0.95), sign=-1.0)
        else:
            _run(threshold,  (0.95, 0.55, 0.20), sign=+1.0)

        if not chunks:
            return None

        mesh = torch.cat(chunks, dim=0)
        if mesh.shape[0] > self.max_vertices:
            # Stratified subsample to stay within the budget.
            n_tris = mesh.shape[0] // 3
            keep_tris = max(1, self.max_vertices // 3)
            sel = torch.randperm(n_tris, device=mesh.device)[:keep_tris]
            tri_starts = sel * 3
            idx = torch.stack([tri_starts, tri_starts + 1, tri_starts + 2], dim=1).reshape(-1)
            mesh = mesh.index_select(0, idx)

        # Persistent CUDA buffer of the right size — the GL hook reads from
        # this pointer asynchronously, so we must keep the tensor alive.
        if (self._mesh_buffer is None
                or self._mesh_buffer.shape != mesh.shape
                or self._mesh_buffer.dtype != mesh.dtype):
            self._mesh_buffer = torch.empty_like(mesh)
        self._mesh_buffer.copy_(mesh)
        return self._mesh_buffer

    def _run_mc(self, field, level, spacing, origin, sdf):
        """Run the selected MC backend; return (vert_grid_idx, world_pos)
        as a pair of (V, 3) tensors, or None.  ``vert_grid_idx`` are
        integer indices into the (cropped) field for gradient lookup.
        """
        dx, dy, dz = spacing
        if self._mc_name == "torchmcubes":
            # torchmcubes returns (V, 3) verts, (F, 3) faces, in voxel coords (grid indices).
            verts, faces = self._mc_callable(field.contiguous(), float(level))
            if verts.numel() == 0 or faces.numel() == 0:
                return None
            tri = verts[faces.reshape(-1).long()]    # (F*3, 3) voxel coords
            world = tri.clone()
            world[:, 0] = world[:, 0] * dx + origin[0]
            world[:, 1] = world[:, 1] * dy + origin[1]
            world[:, 2] = world[:, 2] * dz + origin[2]
            idx = tri.round().long()
            if sdf is not None:
                # Drop triangles whose centroid sits inside the body.
                centroids = world.view(-1, 3, 3).mean(dim=1)
                cent_idx = ((centroids - origin) / torch.tensor(
                    [dx, dy, dz], device=centroids.device, dtype=centroids.dtype
                )).round().long()
                Nx, Ny, Nz = field.shape
                cent_idx[:, 0].clamp_(0, Nx - 1)
                cent_idx[:, 1].clamp_(0, Ny - 1)
                cent_idx[:, 2].clamp_(0, Nz - 1)
                m = sdf[cent_idx[:, 0], cent_idx[:, 1], cent_idx[:, 2]] > 0
                if not m.any():
                    return None
                keep = m.nonzero(as_tuple=True)[0]
                tri_starts = keep * 3
                sel = torch.stack(
                    [tri_starts, tri_starts + 1, tri_starts + 2], dim=1
                ).reshape(-1)
                world = world.index_select(0, sel)
                idx = idx.index_select(0, sel)
            idx[:, 0].clamp_(0, field.shape[0] - 1)
            idx[:, 1].clamp_(0, field.shape[1] - 1)
            idx[:, 2].clamp_(0, field.shape[2] - 1)
            return idx, world

        if self._mc_name == "skimage":
            field_np = field.detach().cpu().numpy()
            try:
                v, f, _, _ = self._mc_callable(field_np, level=float(level), spacing=spacing)
            except (ValueError, RuntimeError):
                return None
            if len(v) == 0 or len(f) == 0:
                return None
            tri_world = v[f.reshape(-1)]            # (F*3, 3) in world units already (spacing applied)
            tri_world = tri_world + origin.detach().cpu().numpy()
            tri_idx = np.round(
                (tri_world - origin.detach().cpu().numpy()) /
                np.array([dx, dy, dz], dtype=np.float32)
            ).astype(np.int64)
            if sdf is not None:
                sdf_np = sdf.detach().cpu().numpy()
                cent_world = tri_world.reshape(-1, 3, 3).mean(axis=1)
                cent_idx = np.round(
                    (cent_world - origin.detach().cpu().numpy()) /
                    np.array([dx, dy, dz], dtype=np.float32)
                ).astype(np.int64)
                Nx, Ny, Nz = field.shape
                cent_idx[:, 0] = np.clip(cent_idx[:, 0], 0, Nx - 1)
                cent_idx[:, 1] = np.clip(cent_idx[:, 1], 0, Ny - 1)
                cent_idx[:, 2] = np.clip(cent_idx[:, 2], 0, Nz - 1)
                m = sdf_np[cent_idx[:, 0], cent_idx[:, 1], cent_idx[:, 2]] > 0
                if not m.any():
                    return None
                keep = np.nonzero(m)[0]
                tri_starts = keep * 3
                sel = np.concatenate([tri_starts, tri_starts + 1, tri_starts + 2])
                sel = sel.reshape(3, -1).T.reshape(-1)
                tri_world = tri_world[sel]
                tri_idx = tri_idx[sel]
            tri_idx[:, 0] = np.clip(tri_idx[:, 0], 0, field.shape[0] - 1)
            tri_idx[:, 1] = np.clip(tri_idx[:, 1], 0, field.shape[1] - 1)
            tri_idx[:, 2] = np.clip(tri_idx[:, 2], 0, field.shape[2] - 1)
            tw = torch.from_numpy(tri_world.astype(np.float32)).to(field.device)
            ti = torch.from_numpy(tri_idx).to(field.device)
            return ti, tw

        return None


