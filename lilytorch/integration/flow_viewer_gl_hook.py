"""Passive-viewer GL hook for the 2-D flow overlay.

The C source is embedded below as a Python string and compiled on first use
(requires gcc).  No separate .c file is needed in the repository.

The compiled shared library is loaded via ``LD_PRELOAD`` into the FARMS
subprocess so that it can intercept ``mjr_render`` at the dynamic-linker
level — the only reliable way to inject an OpenGL overlay into MuJoCo's
passive viewer, whose rendering loop lives entirely in C/C++.

Usage (from your gen_configs script or BaseSimConfig.single_run)::

    from lilytorch.integration.flow_viewer_gl_hook import prepare_mujoco_gl_hook_env
    subprocess.run(['bash', 'run.sh'], env=prepare_mujoco_gl_hook_env())

Inside the simulation the FlowViewer2DGPU extension calls
``get_flow_viewer_gl_hook()`` which returns the singleton controller.
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
#include <string.h>

#define CUDA_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD 2U
#define CUDA_MEMCPY_DEVICE_TO_DEVICE 3
#define CUDA_STREAM_NON_BLOCKING 0x01U
#define CUDA_EVENT_DISABLE_TIMING 0x02U

typedef int cudaError_t;
typedef void* cudaGraphicsResource_t;
typedef void* cudaStream_t;
typedef void* cudaEvent_t;

extern cudaError_t cudaSetDevice(int device);
extern cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int flags);
extern cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags);
extern cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
extern cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags);
extern cudaError_t cudaMemcpyAsync(
    void* dst, const void* src, size_t count, int kind, cudaStream_t stream);
extern cudaError_t cudaGraphicsGLRegisterBuffer(
  cudaGraphicsResource_t* resource,
  unsigned int buffer,
  unsigned int flags);
extern cudaError_t cudaGraphicsMapResources(
    int count,
    cudaGraphicsResource_t* resources,
    void* stream);
extern cudaError_t cudaGraphicsResourceGetMappedPointer(
  void** devPtr,
  size_t* size,
  cudaGraphicsResource_t resource);
extern cudaError_t cudaGraphicsUnmapResources(
    int count,
    cudaGraphicsResource_t* resources,
    void* stream);
extern cudaError_t cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource);
extern cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, int kind);
extern const char* cudaGetErrorString(cudaError_t error);

typedef void (*mjr_render_fn)(mjrRect viewport, mjvScene* scn, const mjrContext* con);

typedef struct {
  pthread_mutex_t mutex;
  int enabled;
  int width;
  int height;
  int device;
  int resource_width;
  int resource_height;
  int resource_device;
  int generation;
  int uploaded_generation;
  uintptr_t cuda_ptr;
  float plane_center[3];
  float plane_size[2];
  float alpha;
  GLuint texture;
  GLuint pbo;
  GLuint vao;
  GLuint vbo;
  GLuint program;
  GLint alpha_location;
  GLint texture_location;
  cudaGraphicsResource_t resource;
  cudaStream_t stream;
  cudaEvent_t ready_event;
  int gl_ready;
} FlowViewerHookState;

static FlowViewerHookState g_state = {
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .enabled = 0,
    .width = 0,
    .height = 0,
    .device = 0,
  .resource_width = 0,
  .resource_height = 0,
  .resource_device = -1,
    .generation = 0,
    .uploaded_generation = -1,
    .cuda_ptr = 0,
    .plane_center = {0.0f, 0.0f, 0.0f},
    .plane_size = {0.0f, 0.0f},
    .alpha = 0.0f,
    .texture = 0,
    .pbo = 0,
    .vao = 0,
    .vbo = 0,
    .program = 0,
    .alpha_location = -1,
    .texture_location = -1,
    .resource = NULL,
    .stream = NULL,
    .ready_event = NULL,
    .gl_ready = 0,
};

static mjr_render_fn real_mjr_render = NULL;
static int logged_cuda_error = 0;
static int logged_gl_error = 0;

static const char* k_vertex_shader =
    "#version 330 core\n"
    "layout (location = 0) in vec3 aClip;\n"
    "layout (location = 1) in vec2 aUv;\n"
    "out vec2 vUv;\n"
    "void main() {\n"
    "  vUv = aUv;\n"
    "  gl_Position = vec4(aClip.xy, 0.0, aClip.z);\n"
    "}\n";

static const char* k_fragment_shader =
    "#version 330 core\n"
    "in vec2 vUv;\n"
    "out vec4 FragColor;\n"
    "uniform sampler2D uTexture;\n"
    "uniform float uAlpha;\n"
    "void main() {\n"
    "  vec3 color = texture(uTexture, vUv).rgb;\n"
    "  vec3 delta = vec3(1.0) - color;\n"
    "  float strength = clamp(max(delta.r, max(delta.g, delta.b)), 0.0, 1.0);\n"
    "  float alpha = uAlpha * (0.30 + 0.70 * pow(strength, 0.4));\n"
    "  FragColor = vec4(color, alpha);\n"
    "}\n";

static void ensure_real_mjr_render(void) {
  if (!real_mjr_render) {
    real_mjr_render = (mjr_render_fn)dlsym(RTLD_NEXT, "mjr_render");
  }
}

static int cuda_ok(cudaError_t error, const char* call) {
  if (error == 0) {
    return 1;
  }
  if (!logged_cuda_error) {
    const char* message = cudaGetErrorString(error);
    fprintf(
        stderr,
        "[FlowViewerGLHook] %s failed with CUDA error %d: %s\n",
        call,
        error,
        message ? message : "unknown");
    logged_cuda_error = 1;
  }
  return 0;
}

static int compile_shader(GLenum shader_type, const char* source, GLuint* shader_out) {
  GLuint shader = glCreateShader(shader_type);
  GLint compiled = 0;
  glShaderSource(shader, 1, &source, NULL);
  glCompileShader(shader);
  glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
  if (!compiled) {
    if (!logged_gl_error) {
      char log_buffer[4096];
      GLsizei log_size = 0;
      glGetShaderInfoLog(shader, (GLsizei)sizeof(log_buffer), &log_size, log_buffer);
      fprintf(stderr, "[FlowViewerGLHook] shader compile failed: %.*s\n", (int)log_size, log_buffer);
      logged_gl_error = 1;
    }
    glDeleteShader(shader);
    return 0;
  }
  *shader_out = shader;
  return 1;
}

static int create_program(GLuint* program_out, GLint* alpha_location, GLint* texture_location) {
  GLuint vertex = 0;
  GLuint fragment = 0;
  GLuint program = 0;
  GLint linked = 0;

  if (!compile_shader(GL_VERTEX_SHADER, k_vertex_shader, &vertex)) {
    return 0;
  }
  if (!compile_shader(GL_FRAGMENT_SHADER, k_fragment_shader, &fragment)) {
    glDeleteShader(vertex);
    return 0;
  }

  program = glCreateProgram();
  glAttachShader(program, vertex);
  glAttachShader(program, fragment);
  glLinkProgram(program);
  glDeleteShader(vertex);
  glDeleteShader(fragment);
  glGetProgramiv(program, GL_LINK_STATUS, &linked);
  if (!linked) {
    if (!logged_gl_error) {
      char log_buffer[4096];
      GLsizei log_size = 0;
      glGetProgramInfoLog(program, (GLsizei)sizeof(log_buffer), &log_size, log_buffer);
      fprintf(stderr, "[FlowViewerGLHook] program link failed: %.*s\n", (int)log_size, log_buffer);
      logged_gl_error = 1;
    }
    glDeleteProgram(program);
    return 0;
  }

  *alpha_location = glGetUniformLocation(program, "uAlpha");
  *texture_location = glGetUniformLocation(program, "uTexture");
  *program_out = program;
  return 1;
}

static void destroy_gl_resources_locked(void) {
  if (g_state.resource) {
    if (g_state.resource_device >= 0) {
      cudaSetDevice(g_state.resource_device);
    }
    cudaGraphicsUnregisterResource(g_state.resource);
    g_state.resource = NULL;
  }
  if (g_state.program) {
    glDeleteProgram(g_state.program);
    g_state.program = 0;
  }
  if (g_state.pbo) {
    glDeleteBuffers(1, &g_state.pbo);
    g_state.pbo = 0;
  }
  if (g_state.vbo) {
    glDeleteBuffers(1, &g_state.vbo);
    g_state.vbo = 0;
  }
  if (g_state.vao) {
    glDeleteVertexArrays(1, &g_state.vao);
    g_state.vao = 0;
  }
  if (g_state.texture) {
    glDeleteTextures(1, &g_state.texture);
    g_state.texture = 0;
  }
  g_state.gl_ready = 0;
  g_state.uploaded_generation = -1;
  g_state.resource_width = 0;
  g_state.resource_height = 0;
  g_state.resource_device = -1;
}

static int ensure_gl_resources_locked(void) {
  GLsizeiptr vertex_bytes = (GLsizeiptr)(4 * 5 * (int)sizeof(float));

  if (g_state.gl_ready &&
      g_state.resource_width == g_state.width &&
      g_state.resource_height == g_state.height &&
      g_state.resource_device == g_state.device) {
    return 1;
  }
  if (g_state.gl_ready) {
    destroy_gl_resources_locked();
  }

  glGenTextures(1, &g_state.texture);
  glBindTexture(GL_TEXTURE_2D, g_state.texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexImage2D(
      GL_TEXTURE_2D,
      0,
      GL_RGB8,
      g_state.width,
      g_state.height,
      0,
      GL_RGB,
      GL_UNSIGNED_BYTE,
      NULL);
  glBindTexture(GL_TEXTURE_2D, 0);

  glGenBuffers(1, &g_state.pbo);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, g_state.pbo);
  glBufferData(
      GL_PIXEL_UNPACK_BUFFER,
      (GLsizeiptr)(g_state.width * g_state.height * 3),
      NULL,
      GL_STREAM_DRAW);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

  if (!cuda_ok(cudaSetDevice(g_state.device), "cudaSetDevice")) {
    destroy_gl_resources_locked();
    return 0;
  }

  if (!cuda_ok(cudaGraphicsGLRegisterBuffer(
          &g_state.resource,
          g_state.pbo,
          CUDA_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD),
        "cudaGraphicsGLRegisterBuffer")) {
    destroy_gl_resources_locked();
    return 0;
  }

  if (!create_program(&g_state.program, &g_state.alpha_location, &g_state.texture_location)) {
    destroy_gl_resources_locked();
    return 0;
  }

  glGenVertexArrays(1, &g_state.vao);
  glGenBuffers(1, &g_state.vbo);
  glBindVertexArray(g_state.vao);
  glBindBuffer(GL_ARRAY_BUFFER, g_state.vbo);
  glBufferData(GL_ARRAY_BUFFER, vertex_bytes, NULL, GL_DYNAMIC_DRAW);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * (GLsizei)sizeof(float), (const void*)0);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * (GLsizei)sizeof(float), (const void*)(3 * sizeof(float)));
  glEnableVertexAttribArray(1);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);

  g_state.resource_width = g_state.width;
  g_state.resource_height = g_state.height;
  g_state.resource_device = g_state.device;
  g_state.gl_ready = 1;
  return 1;
}

static int upload_texture_locked(void) {
  void* mapped_ptr = NULL;
  size_t mapped_size = 0;
  size_t num_bytes = (size_t)(g_state.width * g_state.height * 3);

  if (g_state.generation == g_state.uploaded_generation) {
    return 1;
  }
  if (!g_state.cuda_ptr) {
    return 0;
  }
  if (!ensure_gl_resources_locked()) {
    return 0;
  }
  if (!cuda_ok(cudaSetDevice(g_state.device), "cudaSetDevice")) {
    return 0;
  }
  // Use a dedicated non-blocking stream rather than the legacy default (NULL)
  // stream: the default stream is process-global and any op on it fails with
  // cudaErrorStreamCaptureImplicit (906) while another thread runs a CUDA graph
  // capture (Warp BC-runner / poisson_cuda_graph), poisoning that capture. A
  // non-blocking stream never creates that implicit dependency.
  if (!g_state.stream) {
    if (!cuda_ok(cudaStreamCreateWithFlags(&g_state.stream, CUDA_STREAM_NON_BLOCKING),
                 "cudaStreamCreateWithFlags")) {
      return 0;
    }
  }
  if (!cuda_ok(cudaGraphicsMapResources(1, &g_state.resource, g_state.stream),
               "cudaGraphicsMapResources")) {
    return 0;
  }
  if (!cuda_ok(cudaGraphicsResourceGetMappedPointer(&mapped_ptr, &mapped_size, g_state.resource),
               "cudaGraphicsResourceGetMappedPointer")) {
    cudaGraphicsUnmapResources(1, &g_state.resource, g_state.stream);
    return 0;
  }
  if (mapped_size < num_bytes) {
    cudaGraphicsUnmapResources(1, &g_state.resource, g_state.stream);
    return 0;
  }
  // Order the copy after the producer (torch colormap) kernel: the sim thread
  // records ready_event on its torch stream in lily_flow_viewer_hook_record_ready
  // after writing the texture. A device-side stream-wait replaces the host sync
  // the legacy NULL-stream cudaMemcpy used to provide implicitly.
  if (g_state.ready_event) {
    if (!cuda_ok(cudaStreamWaitEvent(g_state.stream, g_state.ready_event, 0),
                 "cudaStreamWaitEvent")) {
      cudaGraphicsUnmapResources(1, &g_state.resource, g_state.stream);
      return 0;
    }
  }
  if (!cuda_ok(cudaMemcpyAsync(
               mapped_ptr,
               (const void*)g_state.cuda_ptr,
               num_bytes,
               CUDA_MEMCPY_DEVICE_TO_DEVICE,
               g_state.stream),
               "cudaMemcpyAsync")) {
    cudaGraphicsUnmapResources(1, &g_state.resource, g_state.stream);
    return 0;
  }
  // No host sync needed after unmap: cudaGraphicsUnmapResources(stream)
  // guarantees that all CUDA work issued into that stream before the unmap
  // completes before any subsequently issued graphics work (the
  // glTexSubImage2D below) touches the resource. The wait happens on the
  // GPU queue, so the render thread never stalls on the copy.
  if (!cuda_ok(cudaGraphicsUnmapResources(1, &g_state.resource, g_state.stream),
               "cudaGraphicsUnmapResources")) {
    return 0;
  }

  glBindTexture(GL_TEXTURE_2D, g_state.texture);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, g_state.pbo);
  glTexSubImage2D(
      GL_TEXTURE_2D,
      0,
      0,
      0,
      g_state.width,
      g_state.height,
      GL_RGB,
      GL_UNSIGNED_BYTE,
      0);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
  glBindTexture(GL_TEXTURE_2D, 0);

  g_state.uploaded_generation = g_state.generation;
  return 1;
}

static int build_vertices_locked(const mjvScene* scn, int viewport_width, int viewport_height, float vertices[20]) {
  mjtNum headpos[3] = {0, 0, 0};
  mjtNum forward[3] = {0, 0, 0};
  mjtNum up[3] = {0, 0, 0};
  mjtNum right[3] = {0, 0, 0};
  mjtNum rel[3] = {0, 0, 0};
  float uvs[4][2] = {{0.f, 0.f}, {1.f, 0.f}, {0.f, 1.f}, {1.f, 1.f}};
  mjtNum x0 = (mjtNum)(g_state.plane_center[0] - g_state.plane_size[0]);
  mjtNum x1 = (mjtNum)(g_state.plane_center[0] + g_state.plane_size[0]);
  mjtNum y0 = (mjtNum)(g_state.plane_center[1] - g_state.plane_size[1]);
  mjtNum y1 = (mjtNum)(g_state.plane_center[1] + g_state.plane_size[1]);
  mjtNum z = (mjtNum)g_state.plane_center[2];
  mjtNum corners[4][3] = {{x0, y0, z}, {x1, y0, z}, {x0, y1, z}, {x1, y1, z}};
  mjtNum forward_norm;
  mjtNum up_norm;
  mjtNum right_norm;
  mjtNum frustum_height;
  mjtNum half_height_unit;
  mjtNum half_width_unit;
  int index;

  mjv_cameraInModel(headpos, forward, up, scn);
  forward_norm = sqrt(forward[0] * forward[0] + forward[1] * forward[1] + forward[2] * forward[2]);
  up_norm = sqrt(up[0] * up[0] + up[1] * up[1] + up[2] * up[2]);
  if (forward_norm < 1e-8 || up_norm < 1e-8) {
    return 0;
  }
  forward[0] /= forward_norm;
  forward[1] /= forward_norm;
  forward[2] /= forward_norm;
  up[0] /= up_norm;
  up[1] /= up_norm;
  up[2] /= up_norm;

  right[0] = forward[1] * up[2] - forward[2] * up[1];
  right[1] = forward[2] * up[0] - forward[0] * up[2];
  right[2] = forward[0] * up[1] - forward[1] * up[0];
  right_norm = sqrt(right[0] * right[0] + right[1] * right[1] + right[2] * right[2]);
  if (right_norm < 1e-8) {
    return 0;
  }
  right[0] /= right_norm;
  right[1] /= right_norm;
  right[2] /= right_norm;

  frustum_height = mjv_frustumHeight(scn);
  if (frustum_height <= 0) {
    return 0;
  }
  half_height_unit = 0.5 * frustum_height;
  half_width_unit = half_height_unit * ((mjtNum)viewport_width / (mjtNum)viewport_height);

  for (index = 0; index < 4; ++index) {
    mjtNum depth;
    mjtNum x_cam;
    mjtNum y_cam;

    rel[0] = corners[index][0] - headpos[0];
    rel[1] = corners[index][1] - headpos[1];
    rel[2] = corners[index][2] - headpos[2];
    depth = rel[0] * forward[0] + rel[1] * forward[1] + rel[2] * forward[2];
    if (depth <= 1e-6) {
      return 0;
    }
    x_cam = rel[0] * right[0] + rel[1] * right[1] + rel[2] * right[2];
    y_cam = rel[0] * up[0] + rel[1] * up[1] + rel[2] * up[2];
    vertices[index * 5 + 0] = (float)(x_cam / half_width_unit);
    vertices[index * 5 + 1] = (float)(y_cam / half_height_unit);
    vertices[index * 5 + 2] = (float)depth;
    vertices[index * 5 + 3] = uvs[index][0];
    vertices[index * 5 + 4] = uvs[index][1];
  }

  return 1;
}

static void draw_overlay_locked(const mjvScene* scn, mjrRect viewport) {
  float vertices[20];
  GLboolean depth_test_enabled = glIsEnabled(GL_DEPTH_TEST);
  GLboolean blend_enabled = glIsEnabled(GL_BLEND);

  if (!upload_texture_locked()) {
    return;
  }
  if (!build_vertices_locked(scn, viewport.width, viewport.height, vertices)) {
    return;
  }

  glViewport(viewport.left, viewport.bottom, viewport.width, viewport.height);
  glDisable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glUseProgram(g_state.program);
  glUniform1f(g_state.alpha_location, g_state.alpha);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, g_state.texture);
  glUniform1i(g_state.texture_location, 0);
  glBindVertexArray(g_state.vao);
  glBindBuffer(GL_ARRAY_BUFFER, g_state.vbo);
  glBufferSubData(GL_ARRAY_BUFFER, 0, (GLsizeiptr)sizeof(vertices), vertices);
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);
  glBindTexture(GL_TEXTURE_2D, 0);
  glUseProgram(0);
  if (!blend_enabled) {
    glDisable(GL_BLEND);
  }
  if (depth_test_enabled) {
    glEnable(GL_DEPTH_TEST);
  }
}

__attribute__((visibility("default")))
void lily_flow_viewer_hook_set_enabled(int enabled) {
  pthread_mutex_lock(&g_state.mutex);
  g_state.enabled = enabled ? 1 : 0;
  pthread_mutex_unlock(&g_state.mutex);
}

__attribute__((visibility("default")))
void lily_flow_viewer_hook_set_overlay(
    const float plane_center[3],
    const float plane_size[2],
    float alpha,
    const void* cuda_ptr,
    int width,
    int height,
    int device,
    int generation) {
  pthread_mutex_lock(&g_state.mutex);
  g_state.enabled = 1;
  g_state.width = width;
  g_state.height = height;
  g_state.device = device;
  g_state.generation = generation;
  g_state.cuda_ptr = (uintptr_t)cuda_ptr;
  g_state.alpha = alpha;
  memcpy(g_state.plane_center, plane_center, 3 * sizeof(float));
  memcpy(g_state.plane_size, plane_size, 2 * sizeof(float));
  pthread_mutex_unlock(&g_state.mutex);
}

__attribute__((visibility("default")))
void lily_flow_viewer_hook_record_ready(void* producer_stream) {
  // Called on the SIM thread right after the torch colormap kernel that
  // writes the overlay texture, with that thread's current torch stream.
  // Recording here (under the mutex) serialises against the render thread's
  // cudaStreamWaitEvent, so record/wait never race on the event handle.
  // NOTE: this runs on the sim thread sequentially with any Warp graph
  // capture (captures begin and end inside solver calls), so recording on
  // the legacy default stream is safe — unlike the render thread, it can
  // never be concurrent with a capture.
  pthread_mutex_lock(&g_state.mutex);
  if (!g_state.ready_event) {
    if (!cuda_ok(cudaEventCreateWithFlags(&g_state.ready_event, CUDA_EVENT_DISABLE_TIMING),
                 "cudaEventCreateWithFlags")) {
      g_state.ready_event = NULL;
      pthread_mutex_unlock(&g_state.mutex);
      return;
    }
  }
  cuda_ok(cudaEventRecord(g_state.ready_event, (cudaStream_t)producer_stream),
          "cudaEventRecord");
  pthread_mutex_unlock(&g_state.mutex);
}

__attribute__((visibility("default")))
void lily_flow_viewer_hook_clear(void) {
  pthread_mutex_lock(&g_state.mutex);
  g_state.enabled = 0;
  g_state.cuda_ptr = 0;
  g_state.generation = 0;
  g_state.uploaded_generation = -1;
  pthread_mutex_unlock(&g_state.mutex);
}

__attribute__((visibility("default")))
void mjr_render(mjrRect viewport, mjvScene* scn, const mjrContext* con) {
  ensure_real_mjr_render();
  if (real_mjr_render) {
    real_mjr_render(viewport, scn, con);
  }
  if (!con || !scn || con->currentBuffer != mjFB_WINDOW) {
    return;
  }

  pthread_mutex_lock(&g_state.mutex);
  if (g_state.enabled && g_state.width > 0 && g_state.height > 0) {
    draw_overlay_locked(scn, viewport);
  }
  pthread_mutex_unlock(&g_state.mutex);
}
"""


# ── Build helpers ─────────────────────────────────────────────────────────────

_HOOK_ENV_VAR = "LILYTORCH_FLOW_VIEWER_GL_HOOK_LIB"
_BUILD_LOCK = threading.Lock()
_HOOK_INSTANCE = None


def _locate_cudart() -> Path | None:
    candidates: list[Path] = []

    found = ctypes.util.find_library("cudart")
    if found:
        found_path = Path(found)
        if found_path.is_absolute() and found_path.exists():
            candidates.append(found_path)

    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        home = Path(cuda_home)
        candidates.extend([
            home / "targets" / "x86_64-linux" / "lib" / "libcudart.so",
            home / "lib64" / "libcudart.so",
        ])

    for cuda_root in sorted(Path("/usr/local").glob("cuda*")):
        candidates.extend([
            cuda_root / "targets" / "x86_64-linux" / "lib" / "libcudart.so",
            cuda_root / "lib64" / "libcudart.so",
        ])

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return None


def _find_libmujoco(package_dir: Path) -> Path | None:
    for candidate in sorted(package_dir.glob("libmujoco.so*")):
        if candidate.is_file():
            return candidate
    return None


def ensure_flow_viewer_gl_hook_built() -> str:
    """Compile the embedded C source into a shared library and return its path."""
    with _BUILD_LOCK:
        gcc = shutil.which("gcc")
        if gcc is None:
            raise RuntimeError("gcc is required to build the MuJoCo viewer GL hook.")

        import mujoco

        package_dir = Path(mujoco.__file__).resolve().parent
        include_dir = package_dir / "include"
        libmujoco = _find_libmujoco(package_dir)
        if libmujoco is None:
            raise RuntimeError(
                f"Could not find libmujoco.so in {package_dir}."
            )

        # Cache the compiled library next to this Python file.
        # Recompile whenever this Python file (which contains the C source) changes.
        build_dir = Path(__file__).resolve().parent.parent.parent / "build" / "flow_viewer_gl_hook"
        output_path = build_dir / "libflow_viewer_gl_hook.so"
        source_mtime = Path(__file__).stat().st_mtime

        if output_path.exists() and output_path.stat().st_mtime >= source_mtime:
            return str(output_path)

        build_dir.mkdir(parents=True, exist_ok=True)
        cudart = _locate_cudart()
        if cudart is None:
            raise RuntimeError(
                "Could not locate libcudart.so for building the MuJoCo viewer GL hook."
            )

        # Write embedded C source to a temp file for compilation
        with tempfile.NamedTemporaryFile(
            suffix=".c", delete=False, mode="w", dir=build_dir
        ) as tmp:
            tmp.write(_GL_HOOK_C_SOURCE)
            tmp_c = tmp.name

        try:
            compile_cmd = [
                gcc,
                "-shared", "-fPIC", "-O3", "-std=c11",
                "-Wall", "-Wextra", "-Wno-unused-parameter",
                "-I", str(include_dir),
                f"-Wl,-rpath,{package_dir}",
                "-o", str(output_path),
                tmp_c,
                str(libmujoco),
                "-ldl", "-lGL", "-lpthread",
                f"-Wl,-rpath,{cudart.parent}",
                str(cudart),
            ]
            result = subprocess.run(
                compile_cmd, capture_output=True, text=True, check=False
            )
            if result.returncode != 0:
                raise RuntimeError(
                    "Failed to build flow viewer GL hook:\n"
                    f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
                )
        finally:
            try:
                os.unlink(tmp_c)
            except OSError:
                pass

        return str(output_path)


# ── Public API ────────────────────────────────────────────────────────────────

class FlowViewerGLHook:
    def __init__(self, library_path: str):
        self.library_path = library_path
        self._generation = 0
        self._lib = ctypes.CDLL(library_path, mode=ctypes.RTLD_GLOBAL)
        self._lib.lily_flow_viewer_hook_set_enabled.argtypes = [ctypes.c_int]
        self._lib.lily_flow_viewer_hook_set_enabled.restype = None
        self._lib.lily_flow_viewer_hook_set_overlay.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_float,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self._lib.lily_flow_viewer_hook_set_overlay.restype = None
        self._lib.lily_flow_viewer_hook_clear.argtypes = []
        self._lib.lily_flow_viewer_hook_clear.restype = None
        try:
            self._lib.lily_flow_viewer_hook_record_ready.argtypes = [ctypes.c_void_p]
            self._lib.lily_flow_viewer_hook_record_ready.restype = None
            self._record_ready = self._lib.lily_flow_viewer_hook_record_ready
        except AttributeError:
            # Stale preloaded library without the ready-event export: fall back
            # to unordered publish (pre-event behaviour).
            self._record_ready = None

    def set_enabled(self, enabled: bool):
        self._lib.lily_flow_viewer_hook_set_enabled(1 if enabled else 0)

    def clear(self):
        self._lib.lily_flow_viewer_hook_clear()

    def update(
        self,
        texture_rgb: torch.Tensor,
        plane_center,
        plane_size,
        alpha: float,
        synchronize: bool = True,
    ):
        if not texture_rgb.is_cuda:
            raise RuntimeError("FlowViewerGLHook requires a CUDA tensor.")
        if synchronize:
            torch.cuda.current_stream(texture_rgb.device).synchronize()
        elif self._record_ready is not None:
            # Sync-free ordering: record a ready-event on the producer (torch)
            # stream; the hook's copy stream waits on it device-side before
            # reading the texture, so neither the sim nor the render thread
            # blocks on the host.
            self._record_ready(ctypes.c_void_p(
                torch.cuda.current_stream(texture_rgb.device).cuda_stream))

        center = np.asarray(plane_center, dtype=np.float32)
        size_xy = np.asarray(plane_size, dtype=np.float32)
        if center.shape != (3,):
            raise ValueError(f"Expected plane_center shape (3,), got {center.shape}.")
        if size_xy.shape != (2,):
            raise ValueError(f"Expected plane_size shape (2,), got {size_xy.shape}.")

        self._generation += 1
        self._lib.lily_flow_viewer_hook_set_overlay(
            center.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            size_xy.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_float(float(alpha)),
            ctypes.c_void_p(int(texture_rgb.data_ptr())),
            ctypes.c_int(int(texture_rgb.shape[1])),
            ctypes.c_int(int(texture_rgb.shape[0])),
            ctypes.c_int(int(texture_rgb.device.index or 0)),
            ctypes.c_int(self._generation),
        )


def prepare_mujoco_gl_hook_env(base_env: dict | None = None) -> dict:
    """Return an env dict with LD_PRELOAD set to the compiled GL hook library.

    Call this before launching the FARMS subprocess so the hook is active
    from process startup (required for LD_PRELOAD to intercept mjr_render).
    """
    env = dict(os.environ if base_env is None else base_env)

    # CUDA-OpenGL interop requires the viewer's GL context to live on the same
    # GPU as the CUDA device. On hybrid Linux laptops, GLFW commonly lands on
    # the integrated Mesa stack unless the subprocess is nudged toward the
    # NVIDIA GLX vendor up front.
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
        library_path = ensure_flow_viewer_gl_hook_built()
    except Exception as exc:
        print(f"[FlowViewerGLHook] WARNING: could not build GL hook: {exc}")
        return env

    entries = [e for e in env.get("LD_PRELOAD", "").split(":") if e]
    if library_path not in entries:
        entries.insert(0, library_path)
    env["LD_PRELOAD"] = ":".join(entries)
    env[_HOOK_ENV_VAR] = library_path
    return env


def get_flow_viewer_gl_hook() -> FlowViewerGLHook | None:
    """Return the singleton hook controller if the library was preloaded."""
    global _HOOK_INSTANCE

    if _HOOK_INSTANCE is not None:
        return _HOOK_INSTANCE

    library_path = os.environ.get(_HOOK_ENV_VAR)
    if not library_path or not os.path.exists(library_path):
        return None

    _HOOK_INSTANCE = FlowViewerGLHook(library_path)
    return _HOOK_INSTANCE
