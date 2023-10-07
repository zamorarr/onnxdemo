#include <stdio.h>
#include <assert.h>

#include "onnxruntime_c_api.h"

#define ORT_ABORT_ON_ERROR(expr) \
  do { \
    OrtStatus *onnx_status = (expr); \
    if (onnx_status != NULL) { \
      const char *msg = ort->GetErrorMessage(onnx_status); \
      fprintf(stderr, "%s\n", msg); \
      ort->ReleaseStatus(onnx_status); \
      abort(); \
    } \
  } while (0)

void verify_input_output_count(const OrtApi *ort, OrtSession *session) {
  size_t count;
  ORT_ABORT_ON_ERROR(ort->SessionGetInputCount(session, &count));
  assert(count == 3); // input_ids, attention_mask, token_id_types
  ORT_ABORT_ON_ERROR(ort->SessionGetOutputCount(session, &count));
  assert(count == 2); // last_hidden_state, pooler_output
}

int run_inference(const OrtApi *ort, OrtSession *session, int64_t *tokens, const size_t num_tokens) {
  // create memory info
  OrtMemoryInfo *memory_info;
  ORT_ABORT_ON_ERROR(ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));

  int is_tensor;
  
  // create input tensors
  const int64_t input_shape[2] = {1, num_tokens};
  
  OrtValue *input_ids = NULL;
  ORT_ABORT_ON_ERROR(ort->CreateTensorWithDataAsOrtValue(
    memory_info, 
    tokens,  num_tokens * sizeof(int64_t), // input
    input_shape, sizeof(input_shape) / sizeof(input_shape[0]), // input shape
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
    &input_ids
  ));
  assert(input_ids != NULL);
  ORT_ABORT_ON_ERROR(ort->IsTensor(input_ids, &is_tensor));
  assert(is_tensor);

  int64_t mask[num_tokens];
  for (int i = 0; i < num_tokens; i++) mask[i] = 1;
  OrtValue *attention_mask = NULL;
  ORT_ABORT_ON_ERROR(ort->CreateTensorWithDataAsOrtValue(
    memory_info, 
    mask, num_tokens * sizeof(int64_t), // input
    input_shape, sizeof(input_shape) / sizeof(input_shape[0]), // input shape
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
    &attention_mask
  ));
  assert(attention_mask != NULL);
  ORT_ABORT_ON_ERROR(ort->IsTensor(attention_mask, &is_tensor));
  assert(is_tensor);

  int64_t types[num_tokens];
  for (int i = 0; i < num_tokens; i++) types[i] = 0;
  OrtValue *token_type_ids = NULL;
  ORT_ABORT_ON_ERROR(ort->CreateTensorWithDataAsOrtValue(
    memory_info, 
    types, num_tokens * sizeof(int64_t), // input
    input_shape, sizeof(input_shape) / sizeof(input_shape[0]), // input shape
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
    &token_type_ids
  ));
  assert(token_type_ids != NULL);
  ORT_ABORT_ON_ERROR(ort->IsTensor(token_type_ids, &is_tensor));
  assert(is_tensor);

  ort->ReleaseMemoryInfo(memory_info);


  // run inference
  const char *input_names[] = {"input_ids", "attention_mask", "token_type_ids"};
  const char *output_names[] = {"last_hidden_state", "pooler_output"};
  OrtValue *outputs[2] = {NULL, NULL};

  const OrtValue *inputs[3] = {input_ids, attention_mask, token_type_ids};
  ORT_ABORT_ON_ERROR(ort->Run(session, NULL, input_names, inputs, 3, output_names, 2, outputs));

  assert(outputs != NULL);
  assert(outputs[0] != NULL);
  assert(outputs[1] != NULL);
  
  // extract results
  // print output
  int ret = 0;
  float *last_hidden_state;
  ORT_ABORT_ON_ERROR(ort->GetTensorMutableData(outputs[0], (void**)&last_hidden_state));
  
  OrtTensorTypeAndShapeInfo *shape_info;
  size_t dim_count;
  int64_t dims_hidden_state[3];
  ORT_ABORT_ON_ERROR(ort->GetTensorTypeAndShape(outputs[0], &shape_info));
  ORT_ABORT_ON_ERROR(ort->GetDimensionsCount(shape_info, &dim_count));
  assert(dim_count == 3);

  ORT_ABORT_ON_ERROR(ort->GetDimensions(shape_info, dims_hidden_state, sizeof(dims_hidden_state)/sizeof(dims_hidden_state[0])));
  printf("shape: (%li,%li,%li)\n", dims_hidden_state[0], dims_hidden_state[1], dims_hidden_state[2]);

  printf("data: ");
  for (int batch = 0; batch < dims_hidden_state[0]; batch++) {
    printf("\n[\n  [\n");
    for (int token = 0; token < dims_hidden_state[1]; token++) {
      printf("    [");
      for (int feature = 0; feature < 3; feature++) {
        printf("%0.4e, ", last_hidden_state[batch*dims_hidden_state[1]*dims_hidden_state[2] + token*dims_hidden_state[2] + feature]);
      }
      printf("...]\n");
    }
    printf("  ]");
  }
  printf("\n]\n");

  float *pooler_output;
  ORT_ABORT_ON_ERROR(ort->GetTensorMutableData(outputs[1], (void**)&pooler_output));
  ORT_ABORT_ON_ERROR(ort->GetTensorTypeAndShape(outputs[1], &shape_info));
  ORT_ABORT_ON_ERROR(ort->GetDimensionsCount(shape_info, &dim_count));
  assert(dim_count == 2);
  
  int64_t dims[2];
  ORT_ABORT_ON_ERROR(ort->GetDimensions(shape_info, dims, sizeof(dims)/sizeof(dims[0])));
  printf("shape: (%li,%li)\n", dims[0], dims[1]);

  printf("data: [");
  for (int i = 0; i < dims[1]; i++) {
    if (i < 10) printf("%0.4e, ", pooler_output[i]);
  }
  printf("...]\n");

  // free memory
  ort->ReleaseValue(input_ids);
  ort->ReleaseValue(attention_mask);
  ort->ReleaseValue(token_type_ids);
  ort->ReleaseTensorTypeAndShapeInfo(shape_info);

  // return
  return ret;
}

int main(int argc, char *argv[]) {
  // pointer to ort api
  const OrtApi *ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

  // model from https://huggingface.co/optimum/sbert-all-MiniLM-L6-with-pooler
  char *model_path;

  if (argc >= 2) {
    model_path = argv[1];
  } else{
    fprintf(stderr, "Must include model path\n");
    exit(EXIT_FAILURE);
  }

  // create ort environment
  OrtEnv *env;
  ORT_ABORT_ON_ERROR(ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env));
  assert(env != NULL);

  // create ort session
  OrtSessionOptions *session_options;
  ORT_ABORT_ON_ERROR(ort->CreateSessionOptions(&session_options));

  OrtSession *session;
  ORT_ABORT_ON_ERROR(ort->CreateSession(env, model_path, session_options, &session));

  // check
  verify_input_output_count(ort, session);

  // run inference
  const size_t num_tokens = 7;
  int64_t *tokens = malloc(num_tokens * sizeof(int64_t));
  //tokens = {101, 2023, 2003, 2019, 2742, 6251, 102};
  tokens[0] = 101;
  tokens[1] = 2023;
  tokens[2] = 2003;
  tokens[3] = 2019;
  tokens[4] = 2742;
  tokens[5] = 6251;
  tokens[6] = 102;
  
  int ret = run_inference(ort, session, tokens, num_tokens);

  // free memory
  ort->ReleaseSessionOptions(session_options);
  ort->ReleaseSession(session);
  ort->ReleaseEnv(env);
  free(tokens);

  // successful exit
  return ret;
}