#include <stdio.h>
#include <stdlib.h>
#include "matrix_and_cl_methods.h"

int main( int argc, char **argv )
{
    // Parse command line arguments and check they are valid
    int nRows, nCols;
    getCmdLineArgs( argc, argv, &nRows, &nCols );

    // Set up OpenCL
    cl_device_id device;
    cl_context context = simpleOpenContext_GPU(&device);

    // Open up a single command queue with the profiling option off (third argument = 0)
    cl_int status;
    cl_command_queue queue = clCreateCommandQueue( context, device, 0, &status );

    // Allocate memory for the matrix
    float *hostMatrix = (float*) malloc( nRows * nCols * sizeof(float) );

    // Fill the matrix with random values and display
    fillMatrix( hostMatrix, nRows, nCols );
    printf( "Original matrix (only top-left shown if too large):\n" );
    displayMatrix( hostMatrix, nRows, nCols );

    // Allocate memory for the host's matrix and the resulting transposed matrix on
    // the device
    cl_mem deviceMatrix = clCreateBuffer( context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nRows * nCols * sizeof(float), hostMatrix, &status );
  	cl_mem deviceTransposed = clCreateBuffer( context, CL_MEM_WRITE_ONLY, nRows * nCols * sizeof(float), NULL, &status );

    int deviceRows = nRows, deviceCols = nCols;

    // Build the kernel code
    cl_kernel kernel = compileKernelFromFile( "transpose.cl", "transpose", context, device );

    // Specify the kernel arguments
    status = clSetKernelArg( kernel, 0, sizeof(cl_mem), &deviceMatrix );
    status = clSetKernelArg( kernel, 1, sizeof(int), &deviceRows );
    status = clSetKernelArg( kernel, 2, sizeof(int), &deviceCols );
    status = clSetKernelArg( kernel, 3, sizeof(cl_mem), &deviceTransposed );

    // Set the global problem size
    size_t indexSpaceSize[1];
    indexSpaceSize[0] = nRows * nCols;

    // Put the kernel onto the command queue letting the compiler determine a suitable
    // work group size automatically
    status = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, indexSpaceSize, NULL, 0, NULL, NULL );

  	if ( status != CL_SUCCESS )
  	{
  		printf( "Failure enqueuing kernel: Error %d.\n", status );
  		return EXIT_FAILURE;
  	}

    // Read the result from the device to the host
    status = clEnqueueReadBuffer( queue, deviceTransposed, CL_TRUE, 0, nRows * nCols * sizeof(float), hostMatrix, 0, NULL, NULL );

  	if ( status != CL_SUCCESS )
  	{
  		printf( "Could not copy device data to host: Error %d.\n", status );
  		return EXIT_FAILURE;
  	}

    // Display the transposed matrix
    printf( "Transposed matrix (only top-left shown if too large):\n" );
    displayMatrix( hostMatrix, nCols, nRows );

    // Release all resources
    clReleaseMemObject( deviceMatrix     );
    clReleaseMemObject( deviceTransposed );
    clReleaseKernel( kernel  );
    clReleaseCommandQueue( queue   );
    clReleaseContext( context );
    free( hostMatrix );

    return EXIT_SUCCESS;
}
