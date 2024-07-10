EXE = main
OS = $(shell uname)

ifeq ($(OS), Linux)
	CC = nvcc -lOpenCL
endif

ifeq ($(OS), Darwin)
	CC = clang -framework OpenCL -DCL_SILENCE_DEPRECATION
endif

all:
	$(CC) -o $(EXE) main.c
