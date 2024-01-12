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
  int32_t Scale_code = arg_type_ids[1];
  int32_t Shift_code = arg_type_ids[2];
  int32_t ScaleShift_code = arg_type_ids[3];
  void* data = (((TVMValue*)args)[0].v_handle);
  void* Scale = (((TVMValue*)args)[1].v_handle);
  void* Shift = (((TVMValue*)args)[2].v_handle);
  void* ScaleShift = (((TVMValue*)args)[3].v_handle);
  void* data_1 = (((DLTensor*)data)[0].data);
  void* default_function_data_shape = (((DLTensor*)data)[0].shape);
  void* default_function_data_strides = (((DLTensor*)data)[0].strides);
  int32_t dev_id = (((DLTensor*)data)[0].device.device_id);
  void* Scale_1 = (((DLTensor*)Scale)[0].data);
  void* default_function_Scale_shape = (((DLTensor*)Scale)[0].shape);
  void* default_function_Scale_strides = (((DLTensor*)Scale)[0].strides);
  void* Shift_1 = (((DLTensor*)Shift)[0].data);
  void* default_function_Shift_shape = (((DLTensor*)Shift)[0].shape);
  void* default_function_Shift_strides = (((DLTensor*)Shift)[0].strides);
  void* ScaleShift_1 = (((DLTensor*)ScaleShift)[0].data);
  void* default_function_ScaleShift_shape = (((DLTensor*)ScaleShift)[0].shape);
  void* default_function_ScaleShift_strides = (((DLTensor*)ScaleShift)[0].strides);
  if (!(default_function_data_strides == NULL)) {
  }
  if (!(default_function_Scale_strides == NULL)) {
  }
  if (!(default_function_Shift_strides == NULL)) {
  }
  if (!(default_function_ScaleShift_strides == NULL)) {
  }
  for (int32_t b_outer_outer_outer_cc_outer_outer_outer_fused_i_outer_outer_outer_fused_j_outer_outer_outer_fused_cb_outer_outer_outer_fused_b_outer_outer_inner_fused_cc_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused_cb_outer_outer_inner_fused_b_outer_inner_fused_cc_outer_inner_fused_i_outer_inner_fused_j_outer_inner_fused = 0; b_outer_outer_outer_cc_outer_outer_outer_fused_i_outer_outer_outer_fused_j_outer_outer_outer_fused_cb_outer_outer_outer_fused_b_outer_outer_inner_fused_cc_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused_cb_outer_outer_inner_fused_b_outer_inner_fused_cc_outer_inner_fused_i_outer_inner_fused_j_outer_inner_fused < 560; ++b_outer_outer_outer_cc_outer_outer_outer_fused_i_outer_outer_outer_fused_j_outer_outer_outer_fused_cb_outer_outer_outer_fused_b_outer_outer_inner_fused_cc_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused_cb_outer_outer_inner_fused_b_outer_inner_fused_cc_outer_inner_fused_i_outer_inner_fused_j_outer_inner_fused) {
    for (int32_t j_inner = 0; j_inner < 4; ++j_inner) {
      for (int32_t cb_inner = 0; cb_inner < 20; ++cb_inner) {
        int32_t cse_var_2 = ((((b_outer_outer_outer_cc_outer_outer_outer_fused_i_outer_outer_outer_fused_j_outer_outer_outer_fused_cb_outer_outer_outer_fused_b_outer_outer_inner_fused_cc_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused_cb_outer_outer_inner_fused_b_outer_inner_fused_cc_outer_inner_fused_i_outer_inner_fused_j_outer_inner_fused % 112) / 56) * 20) + cb_inner);
        int32_t cse_var_1 = (((b_outer_outer_outer_cc_outer_outer_outer_fused_i_outer_outer_outer_fused_j_outer_outer_outer_fused_cb_outer_outer_outer_fused_b_outer_outer_inner_fused_cc_outer_outer_inner_fused_i_outer_outer_inner_fused_j_outer_outer_inner_fused_cb_outer_outer_inner_fused_b_outer_inner_fused_cc_outer_inner_fused_i_outer_inner_fused_j_outer_inner_fused * 80) + (j_inner * 20)) + cb_inner);
        ((float*)ScaleShift_1)[cse_var_1] = ((((float*)data_1)[cse_var_1] * ((float*)Scale_1)[cse_var_2]) + ((float*)Shift_1)[cse_var_2]);
      }
    }
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
