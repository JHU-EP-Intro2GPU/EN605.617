//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//

// ImageFilter2D.cpp
//
//    This example demonstrates performing gaussian filtering on a 2D image using
//    OpenCL
//
//    Requires FreeImage library for image I/O:
//      http://freeimage.sourceforge.net/

#include <iostream>
#include <fstream>
#include <sstream>
#include <string.h>

#include <CL/cl.h>

//#include "FreeImage.h"

///
//  Create an OpenCL context on the first available platform using
//  either a GPU or CPU depending on what is available.
//
cl_context CreateContext()
{
    cl_int errNum;
    cl_uint numPlatforms;
    cl_platform_id firstPlatformId;
    cl_context context = NULL;

    // First, select an OpenCL platform to run on.  For this example, we
    // simply choose the first available platform.  Normally, you would
    // query for all available platforms and select the most appropriate one.
    errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
    if (errNum != CL_SUCCESS || numPlatforms <= 0)
    {
        std::cerr << "Failed to find any OpenCL platforms." << std::endl;
        return NULL;
    }

    // Next, create an OpenCL context on the platform.  Attempt to
    // create a GPU-based context, and if that fails, try to create
    // a CPU-based context.
    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)firstPlatformId,
        0
    };
    context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU,
                                      NULL, NULL, &errNum);
    if (errNum != CL_SUCCESS)
    {
        std::cout << "Could not create GPU context, trying CPU..." << std::endl;
        context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU,
                                          NULL, NULL, &errNum);
        if (errNum != CL_SUCCESS)
        {
            std::cerr << "Failed to create an OpenCL GPU or CPU context." << std::endl;
            return NULL;
        }
    }

    return context;
}

///
//  Create a command queue on the first device available on the
//  context
//
cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device)
{
    cl_int errNum;
    cl_device_id *devices;
    cl_command_queue commandQueue = NULL;
    size_t deviceBufferSize = -1;

    // First get the size of the devices buffer
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed call to clGetContextInfo(...,GL_CONTEXT_DEVICES,...)";
        return NULL;
    }

    if (deviceBufferSize <= 0)
    {
        std::cerr << "No devices available.";
        return NULL;
    }

    // Allocate memory for the devices buffer
    devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed to get device IDs";
        return NULL;
    }

    // In this example, we just choose the first available device.  In a
    // real program, you would likely use all available devices or choose
    // the highest performance device based on OpenCL device queries
    commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);
    if (commandQueue == NULL)
    {
        std::cerr << "Failed to create commandQueue for device 0";
        return NULL;
    }

    *device = devices[0];
    delete [] devices;
    return commandQueue;
}

///
//  Create an OpenCL program from the kernel source file
//
cl_program CreateProgram(cl_context context, cl_device_id device, const char* fileName)
{
    cl_int errNum;
    cl_program program;

    std::ifstream kernelFile(fileName, std::ios::in);
    if (!kernelFile.is_open())
    {
        std::cerr << "Failed to open file for reading: " << fileName << std::endl;
        return NULL;
    }

    std::ostringstream oss;
    oss << kernelFile.rdbuf();

    std::string srcStdStr = oss.str();
    const char *srcStr = srcStdStr.c_str();
    program = clCreateProgramWithSource(context, 1,
                                        (const char**)&srcStr,
                                        NULL, NULL);
    if (program == NULL)
    {
        std::cerr << "Failed to create CL program from source." << std::endl;
        return NULL;
    }

    errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              sizeof(buildLog), buildLog, NULL);

        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
        clReleaseProgram(program);
        return NULL;
    }

    return program;
}


///
//  Cleanup any created OpenCL resources
//
void Cleanup(cl_context context, cl_command_queue commandQueue,
             cl_program program, cl_kernel kernel, cl_mem imageObjects[2],
             cl_sampler sampler)
{
    for (int i = 0; i < 2; i++)
    {
        if (imageObjects[i] != 0)
            clReleaseMemObject(imageObjects[i]);
    }
    if (commandQueue != 0)
        clReleaseCommandQueue(commandQueue);

    if (kernel != 0)
        clReleaseKernel(kernel);

    if (program != 0)
        clReleaseProgram(program);

    if (sampler != 0)
        clReleaseSampler(sampler);

    if (context != 0)
        clReleaseContext(context);

}

cl_mem loadClImage(cl_context context, void *image, cl_mem_flags flags, size_t width, size_t height){
 
 	cl_int errNum;
	size_t rowpitch = 0;

	cl_image_format format;

	format.image_channel_order = CL_RGBA;
	format.image_channel_data_type = CL_UNSIGNED_INT8;
 
	cl_mem myClImage = clCreateImage2D(
						context, 
						flags, 
						&format, 
						width, 
						height, 
     					rowpitch, 
						image, 
						&errNum 
						); 
	
 	if (errNum != CL_SUCCESS){
    	std::cout << "Error in clCreateImage2D" << std::endl;
    }
	return myClImage;
}


///
//  Round up to the nearest multiple of the group size
//
size_t RoundUp(int groupSize, int globalSize)
{
    int r = globalSize % groupSize;
    if(r == 0)
    {
     	return globalSize;
    }
    else
    {
     	return globalSize + groupSize - r;
    }
}


cl_int saveCLImage(char *fileName, cl_mem clImage, void *image2, size_t *origin, size_t *region, cl_command_queue commandQueue, int width, int height)
{
 	cl_int errNum;
 	//collect results
	errNum = clEnqueueReadImage(commandQueue,
									clImage,	//	cl_mem image,
									CL_TRUE, //	cl_bool blocking_read,
									origin,	//	const size_t origin[3],
									region,	//	const size_t region[3],
									0,	//	size_t row_pitch,
									0,	//	size_t slice_pitch,
									image2,	//	void *ptr,
									0,	//	cl_uint num_events_in_wait_list,
									NULL, //	const cl_event *event_wait_list,
									NULL //	cl_event *event)
								);

	if (errNum != CL_SUCCESS){
    	std::cout << "Error in clEnqueueReadImage" << std::endl;
    }
 
 	FILE *nk = fopen(fileName, "wb");
	fwrite(image2, 1, sizeof(8*(width*height*3+54)), nk);
 	return errNum;
}

///
//	main() for HelloBinaryWorld example
//
int main(int argc, char** argv)
{
    cl_context context = 0;
    cl_command_queue commandQueue = 0;
    cl_program program = 0;
    cl_device_id device = 0;
    cl_kernel kernel = 0;
    cl_mem imageObjects[2] = { 0, 0 };
    cl_sampler sampler = 0;
    cl_int errNum;
   	size_t width = 512;
 	size_t height = 512;
	size_t szGlobalWorkSize[] = {width, height};
	size_t szLocalWorkSize[] = {16, 16};

    if (argc != 3)
    {
        std::cerr << "USAGE: " << argv[0] << " <inputImageFile> <outputImageFiles>" << std::endl;
        return 1;
    }

    // Create an OpenCL context on first available platform
    context = CreateContext();
    if (context == NULL)
    {
        std::cerr << "Failed to create OpenCL context." << std::endl;
        return 1;
    }

    // Create a command-queue on the first device available
    // on the created context
    commandQueue = CreateCommandQueue(context, &device);
    if (commandQueue == NULL)
    {
        Cleanup(context, commandQueue, program, kernel, imageObjects, sampler);
        return 1;
    }

    // Make sure the device supports images, otherwise exit
    cl_bool imageSupport = CL_FALSE;
    clGetDeviceInfo(device, CL_DEVICE_IMAGE_SUPPORT, sizeof(cl_bool),
                    &imageSupport, NULL);
    if (imageSupport != CL_TRUE)
    {
        std::cerr << "OpenCL device does not support images." << std::endl;
        Cleanup(context, commandQueue, program, kernel, imageObjects, sampler);
        return 1;
    }

    // Load input image from file and load it into
    // an OpenCL image object
	cl_mem_flags flags = CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR;

	void *image = fopen(argv[1], "rb");
 	image = (void *)malloc(8 * (width*height*3+54));
 
 	imageObjects[0] = loadClImage(context, image, flags, width, height);
    if (imageObjects[0] == 0)
    {
        std::cerr << "Error loading: " << std::string(argv[1]) << std::endl;
        Cleanup(context, commandQueue, program, kernel, imageObjects, sampler);
        return 1;
    }

    // Create ouput image object
 	cl_mem_flags flags2 = CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR;
	void *image2 = fopen(argv[2], "wb");
 	image2 = (void *)malloc(8 * (width*height*3+54));
    imageObjects[1] = loadClImage(context, image2, flags2, width, height);

    // Create sampler for sampling image object
    sampler = clCreateSampler(context,
                              CL_FALSE, // Non-normalized coordinates
                              CL_ADDRESS_CLAMP_TO_EDGE,
                              CL_FILTER_NEAREST,
                              &errNum);

    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error creating CL sampler object." << std::endl;
        Cleanup(context, commandQueue, program, kernel, imageObjects, sampler);
        return 1;
    }

    // Create OpenCL program
    program = CreateProgram(context, device, "test_kernel.cl");
    if (program == NULL)
    {
        Cleanup(context, commandQueue, program, kernel, imageObjects, sampler);
        return 1;
    }

    // Create OpenCL kernel
    kernel = clCreateKernel(program, "copy", NULL);
    if (kernel == NULL)
    {
        std::cerr << "Failed to create kernel" << std::endl;
        Cleanup(context, commandQueue, program, kernel, imageObjects, sampler);
        return 1;
    }

    // Set the kernel arguments
	errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &imageObjects[0]);
	errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &imageObjects[1]);
//    errNum |= clSetKernelArg(kernel, 2, sizeof(cl_sampler), &sampler);
//    errNum |= clSetKernelArg(kernel, 3, sizeof(cl_int), &width);
//    errNum |= clSetKernelArg(kernel, 4, sizeof(cl_int), &height);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error setting kernel arguments." << std::endl;
        Cleanup(context, commandQueue, program, kernel, imageObjects, sampler);
        return 1;
    }

    size_t localWorkSize[2] = { 16, 16 };
    size_t globalWorkSize[2] =  { RoundUp(localWorkSize[0], width),
                                  RoundUp(localWorkSize[1], height) };

    // Queue the kernel up for execution

 	// Launch kernel
	errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, szGlobalWorkSize, szLocalWorkSize, 0, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error queuing kernel for execution." << std::endl;
        Cleanup(context, commandQueue, program, kernel, imageObjects, sampler);
        return 1;
    }

    // Read the output buffer back to the Host
 	//char *buffer = new char [width * height * 4];
    size_t origin[3] = { 0, 0, 0 };
    size_t region[3] = { width, height, 1};
    errNum = clEnqueueReadImage(commandQueue, imageObjects[1], CL_TRUE,
                                origin, region, 0, 0, image2,
                                0, NULL, NULL);
 	
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error reading result buffer." << std::endl;
        Cleanup(context, commandQueue, program, kernel, imageObjects, sampler);
        return 1;
    }
	FILE *nk = fopen(argv[1], "wb");
	fwrite(image2, 1, sizeof(8*(width*height*3+54)), nk);
    std::cout << std::endl;
    std::cout << "Executed program succesfully." << std::endl;

 	//errNum = saveCLImage(argv[2], imageObjects[2], image2, origin, region, commandQueue, width, height);
    // Save the image out to disk
//    if (!SaveImage(argv[2], buffer, width, height))
//    {
//        std::cerr << "Error writing output image: " << argv[2] << std::endl;
//        Cleanup(context, commandQueue, program, kernel, imageObjects, sampler);
//        delete [] buffer;
//        return 1;
//    }

    //delete [] buffer;
    Cleanup(context, commandQueue, program, kernel, imageObjects, sampler);
    return 0;
}
