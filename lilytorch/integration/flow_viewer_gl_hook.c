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

typedef int cudaError_t;
typedef void* cudaGraphicsResource_t;

extern cudaError_t cudaSetDevice(int device);
extern cudaError_t cudaGraphicsGLRegisterBuffer(
    cudaGraphicsResource_t* resource,
    unsigned int buffer,
    unsigned int flags);
extern cudaError_t cudaGraphicsMapResources(
    int count,
    cudaGraphicsResource_t* resources,
    void* stream);
extern cudaError_t cudaGraphicsResourceGetMappedPointer(
    void** dev_ptr,
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
  int gl_ready;
} FlowViewerHookState;

static FlowViewerHookState g_state = {
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .enabled = 0,
    .width = 0,
    .height = 0,
    .device = 0,
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
    "  float alpha = uAlpha * strength;\n"
    "  if (alpha <= 1e-4) { discard; }\n"
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
    cudaGraphicsUnregisterResource(g_state.resource);
    g_state.resource = NULL;
  }
  if (g_state.program) {
    glDeleteProgram(g_state.program);
    g_state.program = 0;
  }
  if (g_state.vbo) {
    glDeleteBuffers(1, &g_state.vbo);
    g_state.vbo = 0;
  }
  if (g_state.vao) {
    glDeleteVertexArrays(1, &g_state.vao);
    g_state.vao = 0;
  }
  if (g_state.pbo) {
    glDeleteBuffers(1, &g_state.pbo);
    g_state.pbo = 0;
  }
  if (g_state.texture) {
    glDeleteTextures(1, &g_state.texture);
    g_state.texture = 0;
  }
  g_state.gl_ready = 0;
  g_state.uploaded_generation = -1;
}

static int ensure_gl_resources_locked(void) {
  GLsizeiptr vertex_bytes = (GLsizeiptr)(4 * 5 * (int)sizeof(float));
  GLsizeiptr texture_bytes = (GLsizeiptr)(g_state.width * g_state.height * 3);

  if (g_state.gl_ready) {
    return 1;
  }

  glGenTextures(1, &g_state.texture);
  glBindTexture(GL_TEXTURE_2D, g_state.texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
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
  glBufferData(GL_PIXEL_UNPACK_BUFFER, texture_bytes, NULL, GL_STREAM_DRAW);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

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

  g_state.gl_ready = 1;
  return 1;
}

static int upload_texture_locked(void) {
  void* mapped_ptr = NULL;
  size_t mapped_size = 0;
  size_t required_bytes = (size_t)(g_state.width * g_state.height * 3);

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
  if (!cuda_ok(cudaGraphicsMapResources(1, &g_state.resource, NULL), "cudaGraphicsMapResources")) {
    return 0;
  }
  if (!cuda_ok(cudaGraphicsResourceGetMappedPointer(&mapped_ptr, &mapped_size, g_state.resource),
               "cudaGraphicsResourceGetMappedPointer")) {
    cudaGraphicsUnmapResources(1, &g_state.resource, NULL);
    return 0;
  }
  if (mapped_size < required_bytes) {
    cudaGraphicsUnmapResources(1, &g_state.resource, NULL);
    return 0;
  }
  if (!cuda_ok(cudaMemcpy(mapped_ptr, (const void*)g_state.cuda_ptr, required_bytes, CUDA_MEMCPY_DEVICE_TO_DEVICE),
               "cudaMemcpy")) {
    cudaGraphicsUnmapResources(1, &g_state.resource, NULL);
    return 0;
  }
  if (!cuda_ok(cudaGraphicsUnmapResources(1, &g_state.resource, NULL), "cudaGraphicsUnmapResources")) {
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
      (const void*)0);
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
  if (g_state.gl_ready && (g_state.width != width || g_state.height != height)) {
    destroy_gl_resources_locked();
  }
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