#include <stdio.h>
#include <malloc.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <Python.h>
#include <pythread.h>
#include <structmember.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <torch/torch.h>
#include <torch/extension.h>
#include "JpegCoder.hpp"

typedef struct
{
    PyObject_HEAD
    JpegCoder* m_handle;
}NvJpeg;


static PyMemberDef NvJpeg_DataMembers[] =
{
        {(char*)"m_handle",   T_OBJECT, offsetof(NvJpeg, m_handle),   0, (char*)"NvJpeg handle ptr"},
        {NULL, 0, 0, 0, NULL}
};

int NvJpeg_init(PyObject *self, PyObject *args, PyObject *kwds) {
  ((NvJpeg*)self)->m_handle = new JpegCoder();
  return 0;
}


static void NvJpeg_Destruct(PyObject* self)
{
    delete (JpegCoder*)(((NvJpeg*)self)->m_handle);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* NvJpeg_Str(PyObject* Self)
{
    return Py_BuildValue("s", "<nvjpeg-torch.nvjpeg>");
}

static PyObject* NvJpeg_Repr(PyObject* Self)
{
    return NvJpeg_Str(Self);
}

static PyObject* NvJpeg_decode(NvJpeg* Self, PyObject* Argvs)
{
    JpegCoder* m_handle = (JpegCoder*)Self->m_handle;
    
    Py_buffer pyBuf;
    unsigned char* jpegData;
    int len;
    if(!PyArg_ParseTuple(Argvs, "y*", &pyBuf)){
        PyErr_SetString(PyExc_ValueError, "Parse the argument FAILED! You should jpegData byte string!");
        return NULL;
    }
    jpegData = (unsigned char*)pyBuf.buf;
    len = pyBuf.len;
    JpegCoderImage* img;
    try{
        m_handle->ensureThread(PyThread_get_thread_ident());
        img = m_handle->decode((const unsigned char*)jpegData, len);
        PyBuffer_Release(&pyBuf);
    }catch(JpegCoderError e){
        PyBuffer_Release(&pyBuf);
        PyErr_Format(PyExc_ValueError, "%s, Code: %d", e.what(), e.code());
        return NULL;
    }

    unsigned char* data = img->buffer();

    npy_intp dims[3] = {(npy_intp)(img->height), (npy_intp)(img->width), 3};
    PyObject* temp = PyArray_SimpleNewFromData(3, dims, NPY_UINT8, data);

    PyArray_ENABLEFLAGS((PyArrayObject*) temp, NPY_ARRAY_OWNDATA);
    delete(img);
    return temp;
}

// Cached torch.Tensor type for faster type checking
static PyObject* torch_tensor_type = NULL;

// Helper function to get torch.Tensor type (lazy initialization)
static PyObject* get_torch_tensor_type() {
    if (torch_tensor_type == NULL) {
        PyObject* torch_module = PyImport_ImportModule("torch");
        if (torch_module != NULL) {
            torch_tensor_type = PyObject_GetAttrString(torch_module, "Tensor");
            Py_DECREF(torch_module);
        }
    }
    return torch_tensor_type;
}

// Helper function to check if an object is a PyTorch tensor
static bool is_torch_tensor(PyObject* obj) {
    PyObject* tensor_type = get_torch_tensor_type();
    if (tensor_type == NULL) {
        PyErr_Clear();  // Clear any import errors
        return false;
    }
    return PyObject_IsInstance(obj, tensor_type) == 1;
}

// Helper function to convert PyTorch tensor (CHW float/uint8) to HWC uint8 numpy array for encoding
// Supports:
//   - float tensors in [0, 1] range (multiplied by 255)
//   - float tensors in [-1, 1] range (shifted to [0, 1] then multiplied by 255)
//   - uint8 tensors (used directly)
static PyArrayObject* torch_tensor_to_hwc_uint8(PyObject* tensor_obj) {
    torch::Tensor tensor = THPVariable_Unpack(tensor_obj);
    
    // Check tensor dimensions - should be CHW (3D) or NCHW (4D with N=1)
    int ndim = tensor.dim();
    if (ndim == 4) {
        if (tensor.size(0) != 1) {
            PyErr_SetString(PyExc_ValueError, "For 4D tensors (NCHW), batch size must be 1");
            return NULL;
        }
        tensor = tensor.squeeze(0);  // Remove batch dimension
        ndim = 3;
    }
    
    if (ndim != 3) {
        PyErr_SetString(PyExc_ValueError, "Tensor must be 3D (CHW) or 4D (NCHW with N=1)");
        return NULL;
    }
    
    int64_t C = tensor.size(0);
    int64_t H = tensor.size(1);
    int64_t W = tensor.size(2);
    
    if (C != 3) {
        PyErr_SetString(PyExc_ValueError, "Tensor must have 3 channels (CHW format)");
        return NULL;
    }
    
    // Move tensor to CPU if it's on GPU
    if (tensor.is_cuda()) {
        tensor = tensor.cpu();
    }
    
    // Make tensor contiguous
    tensor = tensor.contiguous();
    
    // Convert to uint8 based on dtype
    torch::Tensor uint8_tensor;
    if (tensor.is_floating_point()) {
        // Supports all floating point types: float16, float32, float64, bfloat16, float8_e4m3fn, float8_e5m2, etc.
        // Convert to float32 for processing
        // tensor = tensor.to(torch::kFloat32);
        
        // Detect range: check min value
        float min_val = tensor.min().item<float>();
        
        // If min is negative, assume [-1, 1] range
        if (min_val < 0) {
            // [-1, 1] -> [0, 1]
            tensor = (tensor + 1.0f) / 2.0f;
        }
        // Now tensor is in [0, 1], scale to [0, 255]
        tensor = tensor.clamp(0.0f, 1.0f) * 255.0f;
        uint8_tensor = tensor.to(torch::kUInt8);
    } else if (tensor.dtype() == torch::kUInt8) {
        uint8_tensor = tensor;
    } else {
        PyErr_SetString(PyExc_ValueError, "Tensor dtype must be floating point (float16, float32, float64, bfloat16, float8, etc.) or uint8");
        return NULL;
    }
    
    // Permute from CHW to HWC and convert RGB to BGR
    // PyTorch/torchvision uses RGB, but nvjpeg encoder expects BGR (like OpenCV)
    uint8_tensor = uint8_tensor.permute({1, 2, 0}).contiguous();
    uint8_tensor = uint8_tensor.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::tensor({2, 1, 0})}).contiguous();
    
    // Create numpy array from tensor data
    npy_intp dims[3] = {(npy_intp)H, (npy_intp)W, 3};
    PyObject* arr = PyArray_SimpleNew(3, dims, NPY_UINT8);
    if (arr == NULL) {
        return NULL;
    }
    
    // Copy data to numpy array
    memcpy(PyArray_DATA((PyArrayObject*)arr), uint8_tensor.data_ptr<uint8_t>(), H * W * 3);
    
    return (PyArrayObject*)arr;
}

static PyObject* NvJpeg_encode(NvJpeg* Self, PyObject* Argvs)
{
    PyObject* input_obj;
    unsigned int quality = 70;
    
    // Parse arguments - accept any object and optional quality
    if (!PyArg_ParseTuple(Argvs, "O|I", &input_obj, &quality)){
        PyErr_SetString(PyExc_ValueError, "Parse the argument FAILED! You should pass BGR image (numpy array HWC or torch tensor CHW)!");
        return NULL;
    }

    if (NULL == input_obj){
        Py_INCREF(Py_None);
        return Py_None;
    }

    if(quality > 100){
        quality = 100;
    }

    PyArrayObject *vecin = NULL;
    bool need_decref = false;
    
    // Check if input is a torch tensor
    if (is_torch_tensor(input_obj)) {
        vecin = torch_tensor_to_hwc_uint8(input_obj);
        if (vecin == NULL) {
            return NULL;  // Error already set
        }
        need_decref = true;
    } else if (PyArray_Check(input_obj)) {
        vecin = (PyArrayObject*)input_obj;
        
        if (PyArray_NDIM(vecin) != 3){
            PyErr_SetString(PyExc_ValueError, "Parse the argument FAILED! Numpy array must be height*width*channel (HWC format)!");
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_ValueError, "Input must be a numpy array (HWC) or torch tensor (CHW)!");
        return NULL;
    }

    JpegCoder* m_handle = (JpegCoder*)Self->m_handle;

    PyObject* bytes = PyObject_CallMethod((PyObject*)vecin, "tobytes", NULL);

    Py_buffer pyBuf;

    unsigned char* buffer;
    PyArg_Parse(bytes, "y*", &pyBuf);
    buffer = (unsigned char*)pyBuf.buf;
    auto img = new JpegCoderImage(PyArray_DIM(vecin, 1), PyArray_DIM(vecin, 0), 3, JPEGCODER_CSS_444);
    img->fill(buffer);
    PyBuffer_Release(&pyBuf);
    Py_DECREF(bytes);
    
    if (need_decref) {
        Py_DECREF(vecin);
    }

    m_handle->ensureThread(PyThread_get_thread_ident());
    auto data = m_handle->encode(img, quality);

    PyObject* rtn = PyBytes_FromStringAndSize((const char*)data->data, data->size);

    delete(data);
    delete(img);
    
    return rtn;
}

static PyObject* NvJpeg_read(NvJpeg* Self, PyObject* Argvs)
{
    JpegCoder* m_handle = (JpegCoder*)Self->m_handle;
    
    unsigned char* jpegFile;
    if(!PyArg_ParseTuple(Argvs, "s", &jpegFile)){
        PyErr_SetString(PyExc_ValueError, "Parse the argument FAILED! You should pass jpeg file path string!");
        return NULL;
    }
    #ifdef _MSC_VER
    FILE* fp;
    fopen_s(&fp, (const char*)jpegFile, "rb");
    #else
    FILE* fp = fopen((const char*)jpegFile, "rb");
    #endif

    if (fp == NULL){
        PyErr_Format(PyExc_IOError, "Cannot open file \"%s\"", jpegFile);
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    size_t dataLength = ftell(fp);
    unsigned char* jpegData = (unsigned char*)malloc(dataLength);
    if(jpegData == NULL){
        fclose(fp);
        PyErr_Format(PyExc_IOError, "Out of memeroy when read file \"%s\"", jpegFile);
        return NULL;
    }

    fseek(fp, 0, SEEK_SET);
    if(fread(jpegData, 1, dataLength, fp) != dataLength){
        fclose(fp);
        free(jpegData);
        PyErr_Format(PyExc_IOError, "Read file \"%s\" with error", jpegFile);
        return NULL;
    }

    fclose(fp);

    m_handle->ensureThread(PyThread_get_thread_ident());
    auto img = m_handle->decode((const unsigned char*)jpegData, dataLength);

    free(jpegData);

    unsigned char* data = img->buffer();

    npy_intp dims[3] = {(npy_intp)(img->height), (npy_intp)(img->width), 3};
    PyObject* temp = PyArray_SimpleNewFromData(3, dims, NPY_UINT8, data);

    PyArray_ENABLEFLAGS((PyArrayObject*) temp, NPY_ARRAY_OWNDATA);
    delete(img);
    return temp;
}

static PyObject* NvJpeg_write(NvJpeg* Self, PyObject* Argvs)
{
    unsigned char* jpegFile;
    PyObject* input_obj;
    unsigned int quality = 70;
    
    // Parse: path, image (numpy or tensor), optional quality
    if (!PyArg_ParseTuple(Argvs, "sO|I", &jpegFile, &input_obj, &quality)){
        PyErr_SetString(PyExc_ValueError, "Parse the argument FAILED! You should pass jpeg file path and BGR image (numpy array HWC or torch tensor CHW)!");
        return NULL;
    }

    #ifdef _MSC_VER
    FILE* fp;
    fopen_s(&fp, (const char*)jpegFile, "wb");
    #else
    FILE* fp = fopen((const char*)jpegFile, "wb");
    #endif
    
    if(fp == NULL){
        PyErr_Format(PyExc_IOError, "Cannot open file \"%s\"", jpegFile);
        return NULL;
    }
    
    // Build arguments for encode: (input_obj, quality)
    PyObject* encodeArgs = Py_BuildValue("(OI)", input_obj, quality);
    PyObject* encodeResponse = NvJpeg_encode(Self, encodeArgs);
    Py_DECREF(encodeArgs);
    
    if(encodeResponse == NULL){
        fclose(fp);
        return NULL;
    }

    char* jpegData;
    Py_ssize_t jpegDataSize;
    PyBytes_AsStringAndSize(encodeResponse, &jpegData, &jpegDataSize);
    ssize_t write_size = fwrite(jpegData, 1, jpegDataSize, fp);
    if(write_size != jpegDataSize){
        PyErr_Format(PyExc_IOError, "Write file \"%s\" with error", jpegFile);
    }
    Py_DECREF(encodeResponse);
    fclose(fp);
    return Py_BuildValue("l", (long)jpegDataSize);
}


static PyMethodDef NvJpeg_MethodMembers[] =
{
        {"encode",  (PyCFunction)NvJpeg_encode,  METH_VARARGS,  "encode jpeg from numpy array (HWC BGR) or torch tensor (CHW RGB)"},
        {"decode", (PyCFunction)NvJpeg_decode, METH_VARARGS,  "decode jpeg to numpy array (HWC BGR)"},
        {"read", (PyCFunction)NvJpeg_read, METH_VARARGS,  "read jpeg file and decode to numpy array (HWC BGR)"},
        {"write", (PyCFunction)NvJpeg_write, METH_VARARGS,  "encode and write jpeg file from numpy array (HWC BGR) or torch tensor (CHW RGB)"},
        {NULL, NULL, 0, NULL}
};


static PyTypeObject NvJpeg_ClassInfo =
{
        PyVarObject_HEAD_INIT(NULL, 0)
        "nvjpeg.NvJpeg",
        sizeof(NvJpeg),
        0
};


void NvJpeg_module_destroy(void *_){
    JpegCoder::cleanUpEnv();
}

static PyModuleDef ModuleInfo =
{
        PyModuleDef_HEAD_INIT,
        "NvJpeg Module",
        "NvJpeg by Nvjpeg with PyTorch support",
        -1,
        NULL, NULL, NULL, NULL,
        NvJpeg_module_destroy
};

PyMODINIT_FUNC
PyInit__nvjpeg(void) {
    PyObject * pReturn = NULL;

    NvJpeg_ClassInfo.tp_dealloc   = NvJpeg_Destruct;
    NvJpeg_ClassInfo.tp_repr      = NvJpeg_Repr;
    NvJpeg_ClassInfo.tp_str       = NvJpeg_Str;
    NvJpeg_ClassInfo.tp_flags     = Py_TPFLAGS_DEFAULT;
    NvJpeg_ClassInfo.tp_doc       = "NvJpeg Python Objects---Extensioned by nvjpeg with PyTorch support";
    NvJpeg_ClassInfo.tp_weaklistoffset = 0;
    NvJpeg_ClassInfo.tp_methods   = NvJpeg_MethodMembers;
    NvJpeg_ClassInfo.tp_members   = NvJpeg_DataMembers;
    NvJpeg_ClassInfo.tp_dictoffset = 0;
    NvJpeg_ClassInfo.tp_init      = NvJpeg_init;
    NvJpeg_ClassInfo.tp_new = PyType_GenericNew;

    if(PyType_Ready(&NvJpeg_ClassInfo) < 0) 
        return NULL;

    pReturn = PyModule_Create(&ModuleInfo);
    if(pReturn == NULL)
        return NULL;

    Py_INCREF(&ModuleInfo);

    Py_INCREF(&NvJpeg_ClassInfo);
    if (PyModule_AddObject(pReturn, "NvJpeg", (PyObject*)&NvJpeg_ClassInfo) < 0) {
        Py_DECREF(&NvJpeg_ClassInfo);
        Py_DECREF(pReturn);
        return NULL;
    }

    import_array();
    return pReturn;
}
