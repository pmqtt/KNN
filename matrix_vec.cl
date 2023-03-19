/* kernel.cl
 * Matrix multiplication: C = A * B.
 * Device code.
 */

// OpenCL Kernel

__kernel void matrix_vec(__global float* matrix,
          __global float* vector,
          __global float* result,
          int wA, int wB)
{
   int gid = get_global_id(0);
   float sum = 0.0f;
   for (int i = 0; i < wB; i++) {

       sum += matrix[gid * wB + i] * vector[i];
   }
   result[gid] = sum;
}

/*
__kernel void matrix_vector_multiply(__global float* matrix,
                                      __global float* vector,
                                      __global float* result,
                                      int matrix_width,
                                      int matrix_height) {
    int gid = get_global_id(0);
    float sum = 0.0f;
    for (int i = 0; i < matrix_width; i++) {
        sum += matrix[gid * matrix_width + i] * vector[i];
    }
    result[gid] = sum;
}
*/

/*
__kernel void vector_add(global const int *A, global const int *B, global int *C) {

    // Get the index of the current element to be processed
    int i = get_global_id(0);

    // Do the operation
    C[i] = A[i] + B[i];
}
*/