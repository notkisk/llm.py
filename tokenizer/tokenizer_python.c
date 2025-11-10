#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "tokenizer.h"

typedef struct {
    PyObject_HEAD
    Tokenizer* tok;
} PyTokenizer;

static void PyTokenizer_dealloc(PyTokenizer* self);
static PyObject* PyTokenizer_new(PyTypeObject* type, PyObject* args, PyObject* kwds);
static int PyTokenizer_init(PyTokenizer* self, PyObject* args, PyObject* kwds);
static PyObject* PyTokenizer_train(PyTokenizer* self, PyObject* args, PyObject* kwds);
static PyObject* PyTokenizer_encode(PyTokenizer* self, PyObject* args);
static PyObject* PyTokenizer_decode(PyTokenizer* self, PyObject* args);
static PyObject* PyTokenizer_vocab_size(PyTokenizer* self, PyObject* args);

static PyMethodDef PyTokenizer_methods[] = {
    {"train", (PyCFunction)PyTokenizer_train, METH_VARARGS | METH_KEYWORDS, "Train the tokenizer"},
    {"encode", (PyCFunction)PyTokenizer_encode, METH_VARARGS, "Encode text to token IDs"},
    {"decode", (PyCFunction)PyTokenizer_decode, METH_VARARGS, "Decode token IDs to text"},
    {"vocab_size", (PyCFunction)PyTokenizer_vocab_size, METH_NOARGS, "Get vocabulary size"},
    {NULL, NULL, 0, NULL}
};

static PyTypeObject PyTokenizerType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "tokenizer.Tokenizer",
    .tp_doc = "BPE Tokenizer implemented in C",
    .tp_basicsize = sizeof(PyTokenizer),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyTokenizer_new,
    .tp_init = (initproc)PyTokenizer_init,
    .tp_dealloc = (destructor)PyTokenizer_dealloc,
    .tp_methods = PyTokenizer_methods,
};

static void PyTokenizer_dealloc(PyTokenizer* self) {
    if (self->tok) {
        tokenizer_free(self->tok);
        self->tok = NULL;
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* PyTokenizer_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    PyTokenizer* self = (PyTokenizer*)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->tok = tokenizer_new();
        if (!self->tok) {
            Py_DECREF(self);
            return NULL;
        }
    }
    return (PyObject*)self;
}

static int PyTokenizer_init(PyTokenizer* self, PyObject* args, PyObject* kwds) {
    return 0;
}

static PyObject* PyTokenizer_train(PyTokenizer* self, PyObject* args, PyObject* kwds) {
    const char* text;
    Py_ssize_t text_len;
    size_t vocab_size = 1000;
    
    static char* kwlist[] = {"text", "vocab_size", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s#|k", kwlist, &text, &text_len, &vocab_size)) {
        return NULL;
    }
    
    int result = tokenizer_train(self->tok, text, (size_t)text_len, vocab_size);
    if (result < 0) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to train tokenizer");
        return NULL;
    }
    
    Py_RETURN_NONE;
}

static PyObject* PyTokenizer_encode(PyTokenizer* self, PyObject* args) {
    const char* text;
    Py_ssize_t text_len;
    
    if (!PyArg_ParseTuple(args, "s#", &text, &text_len)) {
        return NULL;
    }
    
    size_t max_output_len = (size_t)text_len * 2;
    uint32_t* output = (uint32_t*)malloc(max_output_len * sizeof(uint32_t));
    if (!output) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory");
        return NULL;
    }
    
    size_t output_len = 0;
    int result = tokenizer_encode(self->tok, text, (size_t)text_len, 
                                  output, &output_len, max_output_len);
    
    if (result < 0) {
        free(output);
        PyErr_SetString(PyExc_RuntimeError, "Failed to encode text");
        return NULL;
    }
    
    PyObject* py_list = PyList_New((Py_ssize_t)output_len);
    if (!py_list) {
        free(output);
        return NULL;
    }
    
    for (size_t i = 0; i < output_len; i++) {
        PyObject* py_int = PyLong_FromUnsignedLong(output[i]);
        if (!py_int) {
            Py_DECREF(py_list);
            free(output);
            return NULL;
        }
        PyList_SET_ITEM(py_list, (Py_ssize_t)i, py_int);
    }
    
    free(output);
    return py_list;
}

static PyObject* PyTokenizer_decode(PyTokenizer* self, PyObject* args) {
    PyObject* py_list;
    
    if (!PyArg_ParseTuple(args, "O", &py_list)) {
        return NULL;
    }
    
    if (!PyList_Check(py_list)) {
        PyErr_SetString(PyExc_TypeError, "Expected a list of integers");
        return NULL;
    }
    
    Py_ssize_t list_len = PyList_Size(py_list);
    uint32_t* tokens = (uint32_t*)malloc((size_t)list_len * sizeof(uint32_t));
    if (!tokens) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory");
        return NULL;
    }
    
    for (Py_ssize_t i = 0; i < list_len; i++) {
        PyObject* py_item = PyList_GetItem(py_list, i);
        if (!PyLong_Check(py_item)) {
            free(tokens);
            PyErr_SetString(PyExc_TypeError, "List items must be integers");
            return NULL;
        }
        tokens[i] = (uint32_t)PyLong_AsUnsignedLong(py_item);
    }
    
    size_t max_output_len = (size_t)list_len * 10;
    char* output = (char*)malloc(max_output_len);
    if (!output) {
        free(tokens);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory");
        return NULL;
    }
    
    size_t output_len = 0;
    int result = tokenizer_decode(self->tok, tokens, (size_t)list_len,
                                   output, &output_len, max_output_len);
    
    free(tokens);
    
    if (result < 0) {
        free(output);
        PyErr_SetString(PyExc_RuntimeError, "Failed to decode tokens");
        return NULL;
    }
    
    PyObject* py_str = PyUnicode_FromStringAndSize(output, (Py_ssize_t)output_len);
    free(output);
    return py_str;
}

static PyObject* PyTokenizer_vocab_size(PyTokenizer* self, PyObject* args) {
    size_t size = tokenizer_vocab_size(self->tok);
    return PyLong_FromSize_t(size);
}

static PyModuleDef tokenizer_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "tokenizer",
    .m_doc = "C-based BPE tokenizer",
    .m_size = -1,
};

PyMODINIT_FUNC PyInit_tokenizer(void) {
    PyObject* m;
    
    if (PyType_Ready(&PyTokenizerType) < 0) {
        return NULL;
    }
    
    m = PyModule_Create(&tokenizer_module);
    if (m == NULL) {
        return NULL;
    }
    
    Py_INCREF(&PyTokenizerType);
    if (PyModule_AddObject(m, "Tokenizer", (PyObject*)&PyTokenizerType) < 0) {
        Py_DECREF(&PyTokenizerType);
        Py_DECREF(m);
        return NULL;
    }
    
    return m;
}

