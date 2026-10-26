// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — NumPy output allocator boundary fixture

#define PY_SSIZE_T_CLEAN
#define NPY_TARGET_VERSION NPY_1_22_API_VERSION
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdint.h>
#include <stdlib.h>

/* Arrays retain this handler, so every returned byte-shifted pointer is freed
 * through its matching callback even after the previous handler is restored. */
static void *unaligned_malloc(void *ctx, size_t size)
{
    (void)ctx;
    if (size == SIZE_MAX) {
        return NULL;
    }
    char *base = malloc(size + 1);
    return base == NULL ? NULL : base + 1;
}

static void *unaligned_calloc(void *ctx, size_t count, size_t size)
{
    (void)ctx;
    if (size != 0 && count > (SIZE_MAX - 1) / size) {
        return NULL;
    }
    char *base = calloc(1, count * size + 1);
    return base == NULL ? NULL : base + 1;
}

static void *unaligned_realloc(void *ctx, void *pointer, size_t size)
{
    if (pointer == NULL) {
        return unaligned_malloc(ctx, size);
    }
    if (size == SIZE_MAX) {
        return NULL;
    }
    char *base = realloc((char *)pointer - 1, size + 1);
    return base == NULL ? NULL : base + 1;
}

static void unaligned_free(void *ctx, void *pointer, size_t size)
{
    (void)ctx;
    (void)size;
    if (pointer != NULL) {
        free((char *)pointer - 1);
    }
}

static PyDataMem_Handler handler = {
    .name = "dense_output_unaligned",
    .version = 1,
    .allocator = {
        .ctx = NULL,
        .malloc = unaligned_malloc,
        .calloc = unaligned_calloc,
        .realloc = unaligned_realloc,
        .free = unaligned_free,
    },
};

static PyObject *install(PyObject *self, PyObject *args)
{
    (void)self;
    (void)args;
    PyObject *capsule = PyCapsule_New(&handler, "mem_handler", NULL);
    if (capsule == NULL) {
        return NULL;
    }
    PyObject *previous = PyDataMem_SetHandler(capsule);
    Py_DECREF(capsule);
    return previous;
}

static PyObject *restore(PyObject *self, PyObject *previous)
{
    (void)self;
    if (!PyCapsule_IsValid(previous, "mem_handler")) {
        PyErr_SetString(PyExc_TypeError, "expected the original NumPy memory handler");
        return NULL;
    }
    return PyDataMem_SetHandler(previous);
}

static PyObject *current(PyObject *self, PyObject *args)
{
    (void)self;
    (void)args;
    return PyDataMem_GetHandler();
}

static PyMethodDef methods[] = {
    {"install", install, METH_NOARGS, "Install a NumPy data allocator with byte-offset alignment."},
    {"restore", restore, METH_O, "Restore the exact prior NumPy data memory handler."},
    {"current", current, METH_NOARGS, "Return the active NumPy data memory handler capsule."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "dense_numpy_allocator",
    .m_doc = "Exercise real NumPy output alignment refusals through installed dense methods.",
    .m_size = -1,
    .m_methods = methods,
};

PyMODINIT_FUNC PyInit_dense_numpy_allocator(void);

PyMODINIT_FUNC PyInit_dense_numpy_allocator(void)
{
    if (_import_array() < 0) {
        return NULL;
    }
    return PyModule_Create(&module);
}
