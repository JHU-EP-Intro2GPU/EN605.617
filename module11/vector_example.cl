__kernel void vector_add_kernel(__global const float *a,
						__global const float *b,
						__global float *result)
{
    int gid = get_global_id(0);
	float4 a_4 = (float4)(a[gid*4], a[gid*4+1], a[gid*4+2], a[gid*4+3]);
	float4 b_4 = (float4)(b[gid*4], b[gid*4+1], b[gid*4+2], b[gid*4+3]);
	float4 c_4 = a_4 + b_4;
    result[gid*4] = c_4.x;
    result[gid*4+1] = c_4.y;
    result[gid*4+2] = c_4.z;
    result[gid*4+3] = c_4.w;
}

__kernel void vector_sub_kernel(__global const float *a,
						__global const float *b,
						__global float *result)
{
    int gid = get_global_id(0);
	float4 a_4 = (float4)(a[gid*4], a[gid*4+1], a[gid*4+2], a[gid*4+3]);
	float4 b_4 = (float4)(b[gid*4], b[gid*4+1], b[gid*4+2], b[gid*4+3]);
	float4 c_4 = b_4 - a_4;
    result[gid*4] = c_4.x;
    result[gid*4+1] = c_4.y;
    result[gid*4+2] = c_4.z;
    result[gid*4+3] = c_4.w;
}
