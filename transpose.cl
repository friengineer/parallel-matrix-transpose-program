// Kernel for matrix transposition.
__kernel
void transpose( __global float *deviceMatrix, int deviceRows, int deviceCols, __global float *deviceTransposed )
{
  int gid = get_global_id(0);

  // Calculate the array element's row and column indexes in the 2D matrix
  int i = gid/deviceCols, j = gid%deviceCols;

  // Transpose the array element
  deviceTransposed[(j*deviceRows) + i] = deviceMatrix[gid];
}
