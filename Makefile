# choose compiler
# example override to clang: make tokenizer CC=clang
CC=gcc

# alternatively we can just place these in a relative directoy
# such as ./include and ./lib
INCLUDE_PATH=${HOME}/onnx/onnxruntime-linux-x64-1.15.1/include
LIB_PATH=${HOME}/onnx/onnxruntime-linux-x64-1.15.1/lib

# build tokenizer
.PHONY: onnxdemo
onnxdemo: src/onnxdemo.c
	$(CC) \
	-Wall \
	-o build/onnxdemo \
	-I $(INCLUDE_PATH) \
	-O0 -g \
	src/onnxdemo.c \
	-L $(LIB_PATH) \
	-l onnxruntime \
	-Wl,-rpath $(LIB_PATH)


# to run:
# build/onnxdemo ~/Downloads/model.onnx
# setting rpath also works
# -Wl,-rpath ${HOME}/onnx/onnxruntime-linux-x64-1.15.1/lib \
# or w/o rpath
# # LD_LIBRARY_PATH=${HOME}/onnx/onnxruntime-linux-x64-1.15.1/lib build/onnxdemo ~/Downloads/model.onnx

# clean
clean:
	rm -f build