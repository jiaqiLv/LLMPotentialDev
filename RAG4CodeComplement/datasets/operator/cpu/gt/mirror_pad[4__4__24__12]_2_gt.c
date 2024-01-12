// tvm target: c -keys=cpu 
#define TVM_EXPORTS
#include "tvm/runtime/c_runtime_api.h"
#include "tvm/runtime/c_backend_api.h"
#include <math.h>
#include <stdbool.h>
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t default_function(void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t default_function(void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle) {
  int32_t data_code = arg_type_ids[0];
  int32_t MirrorPadInput_code = arg_type_ids[1];
  void* data = (((TVMValue*)args)[0].v_handle);
  void* MirrorPadInput = (((TVMValue*)args)[1].v_handle);
  void* data_1 = (((DLTensor*)data)[0].data);
  void* default_function_data_shape = (((DLTensor*)data)[0].shape);
  void* default_function_data_strides = (((DLTensor*)data)[0].strides);
  int32_t dev_id = (((DLTensor*)data)[0].device.device_id);
  void* MirrorPadInput_1 = (((DLTensor*)MirrorPadInput)[0].data);
  void* default_function_MirrorPadInput_shape = (((DLTensor*)MirrorPadInput)[0].shape);
  void* default_function_MirrorPadInput_strides = (((DLTensor*)MirrorPadInput)[0].strides);
  if (!(default_function_data_strides == NULL)) {
  }
  if (!(default_function_MirrorPadInput_strides == NULL)) {
  }
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 405; ++i0_i1_fused) {
    int32_t cse_var_2 = (i0_i1_fused / 15);
    int32_t cse_var_1 = (i0_i1_fused % 15);
    ((float*)MirrorPadInput_1)[i0_i1_fused] = ((float*)data_1)[((((375 <= i0_i1_fused) ? (48 - cse_var_2) : ((i0_i1_fused < 15) ? 0 : (cse_var_2 - 1))) * 12) + ((cse_var_1 == 14) ? 11 : ((cse_var_1 < 2) ? (1 - cse_var_1) : (cse_var_1 - 2))))];
  }
  return 0;
}

// CodegenC: NOTE: Auto-generated entry function
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t __tvm_main__(void* args, int* arg_type_ids, int num_args, void* out_ret_value, int* out_ret_tcode, void* resource_handle) {
  return default_function(args, arg_type_ids, num_args, out_ret_value, out_ret_tcode, resource_handle);
}
