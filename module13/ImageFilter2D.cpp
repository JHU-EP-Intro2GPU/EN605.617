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

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "FreeImage.h"

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

///
//  Load an image using the FreeImage library and create an OpenCL
//  image out of it
//
cl_mem LoadImage(cl_context context, char *fileName, int &width, int &height)
{
    FREE_IMAGE_FORMAT format = FreeImage_GetFileType(fileName, 0);
    FIBITMAP* image = FreeImage_Load(format, fileName);

    // Convert to 32-bit image
    FIBITMAP* temp = image;
    image = FreeImage_ConvertTo32Bits(image);
    FreeImage_Unload(temp);

    width = FreeImage_GetWidth(image);
    height = FreeImage_GetHeight(image);

    char *buffer = new char[width * height * 4];
    memcpy(buffer, FreeImage_GetBits(image), width * height * 4);

    FreeImage_Unload(image);

    // Create OpenCL image
    cl_image_format clImageFormat;
    clImageFormat.image_channel_order = CL_RGBA;
    clImageFormat.image_channel_data_type = CL_UNORM_INT8;

    cl_int errNum;
    cl_mem clImage;
    clImage = clCreateImage2D(context,
                            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            &clImageFormat,
                            width,
                            height,
                            0,
                            buffer,
                            &errNum);

    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error creating CL image object" << std::endl;
        return 0;
    }

    return clImage;
}

///
//  Save an image using the FreeImage library
//
bool SaveImage(char *fileName, char *buffer, int width, int height)
{
    FREE_IMAGE_FORMAT format = FreeImage_GetFIFFromFilename(fileName);
    FIBITMAP *image = FreeImage_ConvertFromRawBits((BYTE*)buffer, width,
                        height, width * 4, 32,
                        0xFF000000, 0x00FF0000, 0x0000FF00);
    return (FreeImage_Save(format, image, fileName) == TRUE) ? true : false;
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
    int width, height;
    imageObjects[0] = LoadImage(context, argv[1], width, height);
    if (imageObjects[0] == 0)
    {
        std::cerr << "Error loading: " << std::string(argv[1]) << std::endl;
        Cleanup(context, commandQueue, program, kernel, imageObjects, sampler);
        return 1;
    }

    // Create ouput image object
    cl_image_format clImageFormat;
    clImageFormat.image_channel_order = CL_RGBA;
    clImageFormat.image_channel_data_type = CL_UNORM_INT8;
    imageObjects[1] = clCreateImage2D(context,
                                       CL_MEM_WRITE_ONLY,
                                       &clImageFormat,
                                       width,
                                       height,
                                       0,
                                       NULL,
                                       &errNum);

    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error creating CL output image object." << std::endl;
        Cleanup(context, commandQueue, program, kernel, imageObjects, sampler);
        return 1;
    }


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
    program = CreateProgram(context, device, "ImageFilter2D.cl");
    if (program == NULL)
    {
        Cleanup(context, commandQueue, program, kernel, imageObjects, sampler);
        return 1;
    }

    // Create OpenCL kernel
    kernel = clCreateKernel(program, "gaussian_filter", NULL);
    if (kernel == NULL)
    {
        std::cerr << "Failed to create kernel" << std::endl;
        Cleanup(context, commandQueue, program, kernel, imageObjects, sampler);
        return 1;
    }

    // Set the kernel arguments
    errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &imageObjects[0]);
    errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &imageObjects[1]);
    errNum |= clSetKernelArg(kernel, 2, sizeof(cl_sampler), &sampler);
    errNum |= clSetKernelArg(kernel, 3, sizeof(cl_int), &width);
    errNum |= clSetKernelArg(kernel, 4, sizeof(cl_int), &height);
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
    errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL,
                                    globalWorkSize, localWorkSize,
                                    0, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error queuing kernel for execution." << std::endl;
        Cleanup(context, commandQueue, program, kernel, imageObjects, sampler);
        return 1;
    }

    // Read the output buffer back to the Host
    char *buffer = new char [width * height * 4];
    size_t origin[3] = { 0, 0, 0 };
    size_t region[3] = { width, height, 1};
    errNum = clEnqueueReadImage(commandQueue, imageObjects[1], CL_TRUE,
                                origin, region, 0, 0, buffer,
                                0, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error reading result buffer." << std::endl;
        Cleanup(context, commandQueue, program, kernel, imageObjects, sampler);
        return 1;
    }

    std::cout << std::endl;
    std::cout << "Executed program succesfully." << std::endl;

    //memset(buffer, 0xff, width * height * 4);
    // Save the image out to disk
    if (!SaveImage(argv[2], buffer, width, height))
    {
        std::cerr << "Error writing output image: " << argv[2] << std::endl;
        Cleanup(context, commandQueue, program, kernel, imageObjects, sampler);
        delete [] buffer;
        return 1;
    }

    delete [] buffer;
    Cleanup(context, commandQueue, program, kernel, imageObjects, sampler);
    return 0;
}>

<header class="navbar navbar-fixed-top navbar-gitlab with-horizontal-nav">
<a class="sr-only gl-accessibility" href="#content-body" tabindex="1">Skip to content</a>
<div class="container-fluid">
<div class="header-content">
<button aria-label="Toggle global navigation" class="side-nav-toggle" type="button">
<span class="sr-only">Toggle navigation</span>
<i class="fa fa-bars"></i>
</button>
<button class="navbar-toggle" type="button">
<span class="sr-only">Toggle navigation</span>
<i class="fa fa-ellipsis-v"></i>
</button>
<div class="navbar-collapse collapse">
<ul class="nav navbar-nav">
<li class="hidden-sm hidden-xs">
<div class="has-location-badge search search-form">
<form class="navbar-form" action="/search" accept-charset="UTF-8" method="get"><input name="utf8" type="hidden" value="&#x2713;" /><div class="search-input-container">
<div class="location-badge">This project</div>
<div class="search-input-wrap">
<div class="dropdown" data-url="/search/autocomplete">
<input type="search" name="search" id="search" placeholder="Search" class="search-input dropdown-menu-toggle no-outline js-search-dashboard-options" spellcheck="false" tabindex="1" autocomplete="off" data-toggle="dropdown" data-issues-path="http://absaroka.jhuep.com/dashboard/issues" data-mr-path="http://absaroka.jhuep.com/dashboard/merge_requests" />
<div class="dropdown-menu dropdown-select">
<div class="dropdown-content"><ul>
<li>
<a class="is-focused dropdown-menu-empty-link">
Loading...
</a>
</li>
</ul>
</div><div class="dropdown-loading"><i class="fa fa-spinner fa-spin"></i></div>
</div>
<i class="search-icon"></i>
<i class="clear-icon js-clear-input"></i>
</div>
</div>
</div>
<input type="hidden" name="group_id" id="group_id" class="js-search-group-options" />
<input type="hidden" name="project_id" id="search_project_id" value="68" class="js-search-project-options" data-project-path="intro_to_gpu" data-name="intro_to_gpu" data-issues-path="/EN-605.417.31_FA16/intro_to_gpu/issues" data-mr-path="/EN-605.417.31_FA16/intro_to_gpu/merge_requests" />
<input type="hidden" name="search_code" id="search_code" value="true" />
<input type="hidden" name="repository_ref" id="repository_ref" value="master" />

<div class="search-autocomplete-opts hide" data-autocomplete-path="/search/autocomplete" data-autocomplete-project-id="68" data-autocomplete-project-ref="master"></div>
</form></div>

</li>
<li class="visible-sm visible-xs">
<a title="Search" aria-label="Search" data-toggle="tooltip" data-placement="bottom" data-container="body" href="/search"><i class="fa fa-search"></i>
</a></li>
<li>
<a title="Admin Area" aria-label="Admin Area" data-toggle="tooltip" data-placement="bottom" data-container="body" href="/admin"><i class="fa fa-wrench fa-fw"></i>
</a></li>
<li>
<a title="Todos" aria-label="Todos" data-toggle="tooltip" data-placement="bottom" data-container="body" href="/dashboard/todos"><i class="fa fa-bell fa-fw"></i>
<span class="badge hidden todos-pending-count">
0
</span>
</a></li>
<li class="header-user dropdown">
<a class="header-user-dropdown-toggle" data-toggle="dropdown" href="/cpascal3"><img width="26" height="26" class="header-user-avatar" src="http://www.gravatar.com/avatar/6a7947cd3f7e4419520a1950de17b3be?s=52&amp;d=identicon" alt="6a7947cd3f7e4419520a1950de17b3be?s=52&amp;d=identicon" />
<i class="fa fa-caret-down"></i>
</a><div class="dropdown-menu-nav dropdown-menu-align-right">
<ul>
<li>
<a class="profile-link" aria-label="Profile" data-user="cpascal3" href="/cpascal3">Profile</a>
</li>
<li>
<a aria-label="Profile Settings" href="/profile">Profile Settings</a>
</li>
<li>
<a aria-label="Help" href="/help">Help</a>
</li>
<li class="divider"></li>
<li>
<a class="sign-out-link" aria-label="Sign out" rel="nofollow" data-method="delete" href="/users/sign_out">Sign out</a>
</li>
</ul>
</div>
</li>
</ul>
</div>
<h1 class="title"><span><a href="/EN-605.417.31_FA16">EN-605.417.31_FA16</a></span> / <a class="project-item-select-holder" href="/EN-605.417.31_FA16/intro_to_gpu">intro_to_gpu</a><button name="button" type="button" class="dropdown-toggle-caret js-projects-dropdown-toggle" aria-label="Toggle switch project dropdown" data-target=".js-dropdown-menu-projects" data-toggle="dropdown"><i class="fa fa-chevron-down"></i></button></h1>
<div class="header-logo">
<a class="home" title="Dashboard" id="logo" href="/"><svg width="36" height="36" class="tanuki-logo">
  <path class="tanuki-shape tanuki-left-ear" fill="#e24329" d="M2 14l9.38 9v-9l-4-12.28c-.205-.632-1.176-.632-1.38 0z"/>
  <path class="tanuki-shape tanuki-right-ear" fill="#e24329" d="M34 14l-9.38 9v-9l4-12.28c.205-.632 1.176-.632 1.38 0z"/>
  <path class="tanuki-shape tanuki-nose" fill="#e24329" d="M18,34.38 3,14 33,14 Z"/>
  <path class="tanuki-shape tanuki-left-eye" fill="#fc6d26" d="M18,34.38 11.38,14 2,14 6,25Z"/>
  <path class="tanuki-shape tanuki-right-eye" fill="#fc6d26" d="M18,34.38 24.62,14 34,14 30,25Z"/>
  <path class="tanuki-shape tanuki-left-cheek" fill="#fca326" d="M2 14L.1 20.16c-.18.565 0 1.2.5 1.56l17.42 12.66z"/>
  <path class="tanuki-shape tanuki-right-cheek" fill="#fca326" d="M34 14l1.9 6.16c.18.565 0 1.2-.5 1.56L18 34.38z"/>
</svg>

</a></div>
<div class="js-dropdown-menu-projects">
<div class="dropdown-menu dropdown-select dropdown-menu-projects">
<div class="dropdown-title"><span>Go to a project</span><button class="dropdown-title-button dropdown-menu-close" aria-label="Close" type="button"><i class="fa fa-times dropdown-menu-close-icon"></i></button></div>
<div class="dropdown-input"><input type="search" id="" class="dropdown-input-field" placeholder="Search your projects" autocomplete="off" /><i class="fa fa-search dropdown-input-search"></i><i role="button" class="fa fa-times dropdown-input-clear js-dropdown-input-clear"></i></div>
<div class="dropdown-content"></div>
<div class="dropdown-loading"><i class="fa fa-spinner fa-spin"></i></div>
</div>
</div>

</div>
</div>
</header>

<script>
  var findFileURL = "/EN-605.417.31_FA16/intro_to_gpu/find_file/master";
</script>

<div class="page-with-sidebar">
<div class="sidebar-wrapper nicescroll">
<div class="sidebar-action-buttons">
<div class="nav-header-btn toggle-nav-collapse" title="Open/Close">
<span class="sr-only">Toggle navigation</span>
<i class="fa fa-bars"></i>
</div>
<div class="nav-header-btn pin-nav-btn has-tooltip  js-nav-pin" data-container="body" data-placement="right" title="Pin Navigation">
<span class="sr-only">Toggle navigation pinning</span>
<i class="fa fa-fw fa-thumb-tack"></i>
</div>
</div>
<div class="nav-sidebar">
<ul class="nav">
<li class="active home"><a title="Projects" class="dashboard-shortcuts-projects" href="/dashboard/projects"><span>
Projects
</span>
</a></li><li class=""><a class="dashboard-shortcuts-activity" title="Activity" href="/dashboard/activity"><span>
Activity
</span>
</a></li><li class=""><a title="Groups" href="/dashboard/groups"><span>
Groups
</span>
</a></li><li class=""><a title="Milestones" href="/dashboard/milestones"><span>
Milestones
</span>
</a></li><li class=""><a title="Issues" class="dashboard-shortcuts-issues" href="/dashboard/issues?assignee_id=5"><span>
Issues
<span class="count">0</span>
</span>
</a></li><li class=""><a title="Merge Requests" class="dashboard-shortcuts-merge_requests" href="/dashboard/merge_requests?assignee_id=5"><span>
Merge Requests
<span class="count">0</span>
</span>
</a></li><li class=""><a title="Snippets" href="/dashboard/snippets"><span>
Snippets
</span>
</a></li></ul>
</div>

</div>
<div class="layout-nav">
<div class="container-fluid">
<div class="controls">
<div class="dropdown project-settings-dropdown">
<a class="dropdown-new btn btn-default" data-toggle="dropdown" href="#" id="project-settings-button">
<i class="fa fa-cog"></i>
<i class="fa fa-caret-down"></i>
</a>
<ul class="dropdown-menu dropdown-menu-align-right">
<li class=""><a title="Members" class="team-tab tab" href="/EN-605.417.31_FA16/intro_to_gpu/project_members"><span>
Members
</span>
</a></li><li class=""><a title="Groups" href="/EN-605.417.31_FA16/intro_to_gpu/group_links"><span>
Groups
</span>
</a></li><li class=""><a title="Deploy Keys" href="/EN-605.417.31_FA16/intro_to_gpu/deploy_keys"><span>
Deploy Keys
</span>
</a></li><li class=""><a title="Webhooks" href="/EN-605.417.31_FA16/intro_to_gpu/hooks"><span>
Webhooks
</span>
</a></li><li class=""><a title="Services" href="/EN-605.417.31_FA16/intro_to_gpu/services"><span>
Services
</span>
</a></li><li class=""><a title="Protected Branches" href="/EN-605.417.31_FA16/intro_to_gpu/protected_branches"><span>
Protected Branches
</span>
</a></li>
<li class="divider"></li>
<li>
<a href="/EN-605.417.31_FA16/intro_to_gpu/edit">Edit Project
</a></li>
</ul>
</div>
</div>
<div class="nav-control scrolling-tabs-container">
<div class="fade-left">
<i class="fa fa-angle-left"></i>
</div>
<div class="fade-right">
<i class="fa fa-angle-right"></i>
</div>
<ul class="nav-links scrolling-tabs">
<li class="home"><a title="Project" class="shortcuts-project" href="/EN-605.417.31_FA16/intro_to_gpu"><span>
Project
</span>
</a></li><li class=""><a title="Activity" class="shortcuts-project-activity" href="/EN-605.417.31_FA16/intro_to_gpu/activity"><span>
Activity
</span>
</a></li><li class="active"><a title="Repository" class="shortcuts-tree" href="/EN-605.417.31_FA16/intro_to_gpu/tree/master"><span>
Repository
</span>
</a></li><li class=""><a title="Container Registry" class="shortcuts-container-registry" href="/EN-605.417.31_FA16/intro_to_gpu/container_registry"><span>
Registry
</span>
</a></li><li class=""><a title="Graphs" class="shortcuts-graphs" href="/EN-605.417.31_FA16/intro_to_gpu/graphs/master"><span>
Graphs
</span>
</a></li><li class=""><a title="Issues" class="shortcuts-issues" href="/EN-605.417.31_FA16/intro_to_gpu/issues"><span>
Issues
<span class="badge count issue_counter">0</span>
</span>
</a></li><li class=""><a title="Merge Requests" class="shortcuts-merge_requests" href="/EN-605.417.31_FA16/intro_to_gpu/merge_requests"><span>
Merge Requests
<span class="badge count merge_counter">0</span>
</span>
</a></li><li class=""><a title="Wiki" class="shortcuts-wiki" href="/EN-605.417.31_FA16/intro_to_gpu/wikis/home"><span>
Wiki
</span>
</a></li><li class="hidden">
<a title="Network" class="shortcuts-network" href="/EN-605.417.31_FA16/intro_to_gpu/network/master">Network
</a></li>
<li class="hidden">
<a class="shortcuts-new-issue" href="/EN-605.417.31_FA16/intro_to_gpu/issues/new">Create a new issue
</a></li>
<li class="hidden">
<a title="Commits" class="shortcuts-commits" href="/EN-605.417.31_FA16/intro_to_gpu/commits/master">Commits
</a></li>
<li class="hidden">
<a title="Issue Boards" class="shortcuts-issue-boards" href="/EN-605.417.31_FA16/intro_to_gpu/boards">Issue Boards</a>
</li>
</ul>
</div>

</div>
</div>
<div class="content-wrapper page-with-layout-nav">
<div class="scrolling-tabs-container sub-nav-scroll">
<div class="fade-left">
<i class="fa fa-angle-left"></i>
</div>
<div class="fade-right">
<i class="fa fa-angle-right"></i>
</div>

<div class="nav-links sub-nav scrolling-tabs">
<ul class="container-fluid container-limited">
<li class="active"><a href="/EN-605.417.31_FA16/intro_to_gpu/tree/master">Files
</a></li><li class=""><a href="/EN-605.417.31_FA16/intro_to_gpu/commits/master">Commits
</a></li><li class=""><a href="/EN-605.417.31_FA16/intro_to_gpu/network/master">Network
</a></li><li class=""><a href="/EN-605.417.31_FA16/intro_to_gpu/compare?from=master&amp;to=master">Compare
</a></li><li class=""><a href="/EN-605.417.31_FA16/intro_to_gpu/branches">Branches
</a></li><li class=""><a href="/EN-605.417.31_FA16/intro_to_gpu/tags">Tags
</a></li></ul>
</div>
</div>

<div class="alert-wrapper">


<div class="flash-container flash-container-page">
</div>


</div>
<div class=" ">
<div class="content" id="content-body">

<div class="container-fluid container-limited">

<div class="tree-holder" id="tree-holder">
<div class="nav-block">
<div class="tree-ref-holder">
<form class="project-refs-form" action="/EN-605.417.31_FA16/intro_to_gpu/refs/switch" accept-charset="UTF-8" method="get"><input name="utf8" type="hidden" value="&#x2713;" /><input type="hidden" name="destination" id="destination" value="blob" />
<input type="hidden" name="path" id="path" value="opencl-book-samples/src/Chapter_8/ImageFilter2D/ImageFilter2D.cpp" />
<div class="dropdown">
<button class="dropdown-menu-toggle js-project-refs-dropdown" type="button" data-toggle="dropdown" data-selected="master" data-ref="master" data-refs-url="/EN-605.417.31_FA16/intro_to_gpu/refs" data-field-name="ref" data-submit-form-on-click="true"><span class="dropdown-toggle-text ">master</span><i class="fa fa-chevron-down"></i></button>
<div class="dropdown-menu dropdown-menu-selectable">
<div class="dropdown-title"><span>Switch branch/tag</span><button class="dropdown-title-button dropdown-menu-close" aria-label="Close" type="button"><i class="fa fa-times dropdown-menu-close-icon"></i></button></div>
<div class="dropdown-input"><input type="search" id="" class="dropdown-input-field" placeholder="Search branches and tags" autocomplete="off" /><i class="fa fa-search dropdown-input-search"></i><i role="button" class="fa fa-times dropdown-input-clear js-dropdown-input-clear"></i></div>
<div class="dropdown-content"></div>
<div class="dropdown-loading"><i class="fa fa-spinner fa-spin"></i></div>
</div>
</div>
</form>
</div>
<ul class="breadcrumb repo-breadcrumb">
<li>
<a href="/EN-605.417.31_FA16/intro_to_gpu/tree/master">intro_to_gpu
</a></li>
<li>
<a href="/EN-605.417.31_FA16/intro_to_gpu/tree/master/opencl-book-samples">opencl-book-samples</a>
</li>
<li>
<a href="/EN-605.417.31_FA16/intro_to_gpu/tree/master/opencl-book-samples/src">src</a>
</li>
<li>
<a href="/EN-605.417.31_FA16/intro_to_gpu/tree/master/opencl-book-samples/src/Chapter_8">Chapter_8</a>
</li>
<li>
<a href="/EN-605.417.31_FA16/intro_to_gpu/tree/master/opencl-book-samples/src/Chapter_8/ImageFilter2D">ImageFilter2D</a>
</li>
<li>
<a href="/EN-605.417.31_FA16/intro_to_gpu/blob/master/opencl-book-samples/src/Chapter_8/ImageFilter2D/ImageFilter2D.cpp"><strong>
ImageFilter2D.cpp
</strong>
</a></li>
</ul>
</div>
<ul class="blob-commit-info hidden-xs">
<li class="commit js-toggle-container" id="commit-ca317aee">
<a href="/cpascal3"><img class="avatar has-tooltip s36 hidden-xs" alt="Chancellor Pascale&#39;s avatar" title="Chancellor Pascale" data-container="body" src="http://www.gravatar.com/avatar/6a7947cd3f7e4419520a1950de17b3be?s=72&amp;d=identicon" /></a>
<div class="commit-info-block">
<div class="commit-row-title">
<span class="item-title">
<a class="commit-row-message" href="/EN-605.417.31_FA16/intro_to_gpu/commit/ca317aee57f20e0a2303ed5d7da96d1c279991ac">committing the source code for the book right into the master branch, you might …</a>
<span class="commit-row-message visible-xs-inline">
&middot;
ca317aee
</span>
<a class="text-expander hidden-xs js-toggle-button">...</a>
</span>
<div class="commit-actions hidden-xs">
<button class="btn btn-clipboard btn-transparent" data-toggle="tooltip" data-placement="bottom" data-container="body" data-clipboard-text="ca317aee57f20e0a2303ed5d7da96d1c279991ac" type="button" title="Copy to clipboard"><i class="fa fa-clipboard"></i></button>
<a class="commit-short-id btn btn-transparent" href="/EN-605.417.31_FA16/intro_to_gpu/commit/ca317aee57f20e0a2303ed5d7da96d1c279991ac">ca317aee</a>

</div>
</div>
<pre class="commit-row-description js-toggle-content">…want to create duplicate folder for your specific system since using cmake creates a bunch of files and may be difficult to reconstruct this folder if stuff goes bad.</pre>
<a class="commit-author-link has-tooltip" title="cpascal3@jhu.edu" href="/cpascal3">Chancellor Pascale</a>
committed
<time class="js-timeago" title="Oct 11, 2015 5:53pm" datetime="2015-10-11T17:53:50Z" data-toggle="tooltip" data-placement="top" data-container="body">2015-10-11 13:53:50 -0400</time>
</div>
</li>

</ul>
<div class="blob-content-holder" id="blob-content-holder">
<article class="file-holder">
<div class="file-title">
<i class="fa fa-file-text-o fa-fw"></i>
<strong>
ImageFilter2D.cpp
</strong>
<small>
13.7 KB
</small>
<div class="file-actions hidden-xs">
<div class="btn-group tree-btn-group">
<a class="btn btn-sm" target="_blank" href="/EN-605.417.31_FA16/intro_to_gpu/raw/master/opencl-book-samples/src/Chapter_8/ImageFilter2D/ImageFilter2D.cpp">Raw</a>
<a class="btn btn-sm" href="/EN-605.417.31_FA16/intro_to_gpu/blame/master/opencl-book-samples/src/Chapter_8/ImageFilter2D/ImageFilter2D.cpp">Blame</a>
<a class="btn btn-sm" href="/EN-605.417.31_FA16/intro_to_gpu/commits/master/opencl-book-samples/src/Chapter_8/ImageFilter2D/ImageFilter2D.cpp">History</a>
<a class="btn btn-sm" href="/EN-605.417.31_FA16/intro_to_gpu/blob/18ea4946eebf9c9b710405b0848cb85613d7ac47/opencl-book-samples/src/Chapter_8/ImageFilter2D/ImageFilter2D.cpp">Permalink</a>
</div>
<div class="btn-group" role="group">
<a class="btn btn-sm" href="/EN-605.417.31_FA16/intro_to_gpu/edit/master/opencl-book-samples/src/Chapter_8/ImageFilter2D/ImageFilter2D.cpp">Edit</a>
<button name="button" type="submit" class="btn btn-default" data-target="#modal-upload-blob" data-toggle="modal">Replace</button>
<button name="button" type="submit" class="btn btn-remove" data-target="#modal-remove-blob" data-toggle="modal">Delete</button>
</div>

</div>
</div>
<div class="file-content code js-syntax-highlight">
<div class="line-numbers">
<a class="diff-line-num" data-line-number="1" href="#L1" id="L1">
<i class="fa fa-link"></i>
1
</a>
<a class="diff-line-num" data-line-number="2" href="#L2" id="L2">
<i class="fa fa-link"></i>
2
</a>
<a class="diff-line-num" data-line-number="3" href="#L3" id="L3">
<i class="fa fa-link"></i>
3
</a>
<a class="diff-line-num" data-line-number="4" href="#L4" id="L4">
<i class="fa fa-link"></i>
4
</a>
<a class="diff-line-num" data-line-number="5" href="#L5" id="L5">
<i class="fa fa-link"></i>
5
</a>
<a class="diff-line-num" data-line-number="6" href="#L6" id="L6">
<i class="fa fa-link"></i>
6
</a>
<a class="diff-line-num" data-line-number="7" href="#L7" id="L7">
<i class="fa fa-link"></i>
7
</a>
<a class="diff-line-num" data-line-number="8" href="#L8" id="L8">
<i class="fa fa-link"></i>
8
</a>
<a class="diff-line-num" data-line-number="9" href="#L9" id="L9">
<i class="fa fa-link"></i>
9
</a>
<a class="diff-line-num" data-line-number="10" href="#L10" id="L10">
<i class="fa fa-link"></i>
10
</a>
<a class="diff-line-num" data-line-number="11" href="#L11" id="L11">
<i class="fa fa-link"></i>
11
</a>
<a class="diff-line-num" data-line-number="12" href="#L12" id="L12">
<i class="fa fa-link"></i>
12
</a>
<a class="diff-line-num" data-line-number="13" href="#L13" id="L13">
<i class="fa fa-link"></i>
13
</a>
<a class="diff-line-num" data-line-number="14" href="#L14" id="L14">
<i class="fa fa-link"></i>
14
</a>
<a class="diff-line-num" data-line-number="15" href="#L15" id="L15">
<i class="fa fa-link"></i>
15
</a>
<a class="diff-line-num" data-line-number="16" href="#L16" id="L16">
<i class="fa fa-link"></i>
16
</a>
<a class="diff-line-num" data-line-number="17" href="#L17" id="L17">
<i class="fa fa-link"></i>
17
</a>
<a class="diff-line-num" data-line-number="18" href="#L18" id="L18">
<i class="fa fa-link"></i>
18
</a>
<a class="diff-line-num" data-line-number="19" href="#L19" id="L19">
<i class="fa fa-link"></i>
19
</a>
<a class="diff-line-num" data-line-number="20" href="#L20" id="L20">
<i class="fa fa-link"></i>
20
</a>
<a class="diff-line-num" data-line-number="21" href="#L21" id="L21">
<i class="fa fa-link"></i>
21
</a>
<a class="diff-line-num" data-line-number="22" href="#L22" id="L22">
<i class="fa fa-link"></i>
22
</a>
<a class="diff-line-num" data-line-number="23" href="#L23" id="L23">
<i class="fa fa-link"></i>
23
</a>
<a class="diff-line-num" data-line-number="24" href="#L24" id="L24">
<i class="fa fa-link"></i>
24
</a>
<a class="diff-line-num" data-line-number="25" href="#L25" id="L25">
<i class="fa fa-link"></i>
25
</a>
<a class="diff-line-num" data-line-number="26" href="#L26" id="L26">
<i class="fa fa-link"></i>
26
</a>
<a class="diff-line-num" data-line-number="27" href="#L27" id="L27">
<i class="fa fa-link"></i>
27
</a>
<a class="diff-line-num" data-line-number="28" href="#L28" id="L28">
<i class="fa fa-link"></i>
28
</a>
<a class="diff-line-num" data-line-number="29" href="#L29" id="L29">
<i class="fa fa-link"></i>
29
</a>
<a class="diff-line-num" data-line-number="30" href="#L30" id="L30">
<i class="fa fa-link"></i>
30
</a>
<a class="diff-line-num" data-line-number="31" href="#L31" id="L31">
<i class="fa fa-link"></i>
31
</a>
<a class="diff-line-num" data-line-number="32" href="#L32" id="L32">
<i class="fa fa-link"></i>
32
</a>
<a class="diff-line-num" data-line-number="33" href="#L33" id="L33">
<i class="fa fa-link"></i>
33
</a>
<a class="diff-line-num" data-line-number="34" href="#L34" id="L34">
<i class="fa fa-link"></i>
34
</a>
<a class="diff-line-num" data-line-number="35" href="#L35" id="L35">
<i class="fa fa-link"></i>
35
</a>
<a class="diff-line-num" data-line-number="36" href="#L36" id="L36">
<i class="fa fa-link"></i>
36
</a>
<a class="diff-line-num" data-line-number="37" href="#L37" id="L37">
<i class="fa fa-link"></i>
37
</a>
<a class="diff-line-num" data-line-number="38" href="#L38" id="L38">
<i class="fa fa-link"></i>
38
</a>
<a class="diff-line-num" data-line-number="39" href="#L39" id="L39">
<i class="fa fa-link"></i>
39
</a>
<a class="diff-line-num" data-line-number="40" href="#L40" id="L40">
<i class="fa fa-link"></i>
40
</a>
<a class="diff-line-num" data-line-number="41" href="#L41" id="L41">
<i class="fa fa-link"></i>
41
</a>
<a class="diff-line-num" data-line-number="42" href="#L42" id="L42">
<i class="fa fa-link"></i>
42
</a>
<a class="diff-line-num" data-line-number="43" href="#L43" id="L43">
<i class="fa fa-link"></i>
43
</a>
<a class="diff-line-num" data-line-number="44" href="#L44" id="L44">
<i class="fa fa-link"></i>
44
</a>
<a class="diff-line-num" data-line-number="45" href="#L45" id="L45">
<i class="fa fa-link"></i>
45
</a>
<a class="diff-line-num" data-line-number="46" href="#L46" id="L46">
<i class="fa fa-link"></i>
46
</a>
<a class="diff-line-num" data-line-number="47" href="#L47" id="L47">
<i class="fa fa-link"></i>
47
</a>
<a class="diff-line-num" data-line-number="48" href="#L48" id="L48">
<i class="fa fa-link"></i>
48
</a>
<a class="diff-line-num" data-line-number="49" href="#L49" id="L49">
<i class="fa fa-link"></i>
49
</a>
<a class="diff-line-num" data-line-number="50" href="#L50" id="L50">
<i class="fa fa-link"></i>
50
</a>
<a class="diff-line-num" data-line-number="51" href="#L51" id="L51">
<i class="fa fa-link"></i>
51
</a>
<a class="diff-line-num" data-line-number="52" href="#L52" id="L52">
<i class="fa fa-link"></i>
52
</a>
<a class="diff-line-num" data-line-number="53" href="#L53" id="L53">
<i class="fa fa-link"></i>
53
</a>
<a class="diff-line-num" data-line-number="54" href="#L54" id="L54">
<i class="fa fa-link"></i>
54
</a>
<a class="diff-line-num" data-line-number="55" href="#L55" id="L55">
<i class="fa fa-link"></i>
55
</a>
<a class="diff-line-num" data-line-number="56" href="#L56" id="L56">
<i class="fa fa-link"></i>
56
</a>
<a class="diff-line-num" data-line-number="57" href="#L57" id="L57">
<i class="fa fa-link"></i>
57
</a>
<a class="diff-line-num" data-line-number="58" href="#L58" id="L58">
<i class="fa fa-link"></i>
58
</a>
<a class="diff-line-num" data-line-number="59" href="#L59" id="L59">
<i class="fa fa-link"></i>
59
</a>
<a class="diff-line-num" data-line-number="60" href="#L60" id="L60">
<i class="fa fa-link"></i>
60
</a>
<a class="diff-line-num" data-line-number="61" href="#L61" id="L61">
<i class="fa fa-link"></i>
61
</a>
<a class="diff-line-num" data-line-number="62" href="#L62" id="L62">
<i class="fa fa-link"></i>
62
</a>
<a class="diff-line-num" data-line-number="63" href="#L63" id="L63">
<i class="fa fa-link"></i>
63
</a>
<a class="diff-line-num" data-line-number="64" href="#L64" id="L64">
<i class="fa fa-link"></i>
64
</a>
<a class="diff-line-num" data-line-number="65" href="#L65" id="L65">
<i class="fa fa-link"></i>
65
</a>
<a class="diff-line-num" data-line-number="66" href="#L66" id="L66">
<i class="fa fa-link"></i>
66
</a>
<a class="diff-line-num" data-line-number="67" href="#L67" id="L67">
<i class="fa fa-link"></i>
67
</a>
<a class="diff-line-num" data-line-number="68" href="#L68" id="L68">
<i class="fa fa-link"></i>
68
</a>
<a class="diff-line-num" data-line-number="69" href="#L69" id="L69">
<i class="fa fa-link"></i>
69
</a>
<a class="diff-line-num" data-line-number="70" href="#L70" id="L70">
<i class="fa fa-link"></i>
70
</a>
<a class="diff-line-num" data-line-number="71" href="#L71" id="L71">
<i class="fa fa-link"></i>
71
</a>
<a class="diff-line-num" data-line-number="72" href="#L72" id="L72">
<i class="fa fa-link"></i>
72
</a>
<a class="diff-line-num" data-line-number="73" href="#L73" id="L73">
<i class="fa fa-link"></i>
73
</a>
<a class="diff-line-num" data-line-number="74" href="#L74" id="L74">
<i class="fa fa-link"></i>
74
</a>
<a class="diff-line-num" data-line-number="75" href="#L75" id="L75">
<i class="fa fa-link"></i>
75
</a>
<a class="diff-line-num" data-line-number="76" href="#L76" id="L76">
<i class="fa fa-link"></i>
76
</a>
<a class="diff-line-num" data-line-number="77" href="#L77" id="L77">
<i class="fa fa-link"></i>
77
</a>
<a class="diff-line-num" data-line-number="78" href="#L78" id="L78">
<i class="fa fa-link"></i>
78
</a>
<a class="diff-line-num" data-line-number="79" href="#L79" id="L79">
<i class="fa fa-link"></i>
79
</a>
<a class="diff-line-num" data-line-number="80" href="#L80" id="L80">
<i class="fa fa-link"></i>
80
</a>
<a class="diff-line-num" data-line-number="81" href="#L81" id="L81">
<i class="fa fa-link"></i>
81
</a>
<a class="diff-line-num" data-line-number="82" href="#L82" id="L82">
<i class="fa fa-link"></i>
82
</a>
<a class="diff-line-num" data-line-number="83" href="#L83" id="L83">
<i class="fa fa-link"></i>
83
</a>
<a class="diff-line-num" data-line-number="84" href="#L84" id="L84">
<i class="fa fa-link"></i>
84
</a>
<a class="diff-line-num" data-line-number="85" href="#L85" id="L85">
<i class="fa fa-link"></i>
85
</a>
<a class="diff-line-num" data-line-number="86" href="#L86" id="L86">
<i class="fa fa-link"></i>
86
</a>
<a class="diff-line-num" data-line-number="87" href="#L87" id="L87">
<i class="fa fa-link"></i>
87
</a>
<a class="diff-line-num" data-line-number="88" href="#L88" id="L88">
<i class="fa fa-link"></i>
88
</a>
<a class="diff-line-num" data-line-number="89" href="#L89" id="L89">
<i class="fa fa-link"></i>
89
</a>
<a class="diff-line-num" data-line-number="90" href="#L90" id="L90">
<i class="fa fa-link"></i>
90
</a>
<a class="diff-line-num" data-line-number="91" href="#L91" id="L91">
<i class="fa fa-link"></i>
91
</a>
<a class="diff-line-num" data-line-number="92" href="#L92" id="L92">
<i class="fa fa-link"></i>
92
</a>
<a class="diff-line-num" data-line-number="93" href="#L93" id="L93">
<i class="fa fa-link"></i>
93
</a>
<a class="diff-line-num" data-line-number="94" href="#L94" id="L94">
<i class="fa fa-link"></i>
94
</a>
<a class="diff-line-num" data-line-number="95" href="#L95" id="L95">
<i class="fa fa-link"></i>
95
</a>
<a class="diff-line-num" data-line-number="96" href="#L96" id="L96">
<i class="fa fa-link"></i>
96
</a>
<a class="diff-line-num" data-line-number="97" href="#L97" id="L97">
<i class="fa fa-link"></i>
97
</a>
<a class="diff-line-num" data-line-number="98" href="#L98" id="L98">
<i class="fa fa-link"></i>
98
</a>
<a class="diff-line-num" data-line-number="99" href="#L99" id="L99">
<i class="fa fa-link"></i>
99
</a>
<a class="diff-line-num" data-line-number="100" href="#L100" id="L100">
<i class="fa fa-link"></i>
100
</a>
<a class="diff-line-num" data-line-number="101" href="#L101" id="L101">
<i class="fa fa-link"></i>
101
</a>
<a class="diff-line-num" data-line-number="102" href="#L102" id="L102">
<i class="fa fa-link"></i>
102
</a>
<a class="diff-line-num" data-line-number="103" href="#L103" id="L103">
<i class="fa fa-link"></i>
103
</a>
<a class="diff-line-num" data-line-number="104" href="#L104" id="L104">
<i class="fa fa-link"></i>
104
</a>
<a class="diff-line-num" data-line-number="105" href="#L105" id="L105">
<i class="fa fa-link"></i>
105
</a>
<a class="diff-line-num" data-line-number="106" href="#L106" id="L106">
<i class="fa fa-link"></i>
106
</a>
<a class="diff-line-num" data-line-number="107" href="#L107" id="L107">
<i class="fa fa-link"></i>
107
</a>
<a class="diff-line-num" data-line-number="108" href="#L108" id="L108">
<i class="fa fa-link"></i>
108
</a>
<a class="diff-line-num" data-line-number="109" href="#L109" id="L109">
<i class="fa fa-link"></i>
109
</a>
<a class="diff-line-num" data-line-number="110" href="#L110" id="L110">
<i class="fa fa-link"></i>
110
</a>
<a class="diff-line-num" data-line-number="111" href="#L111" id="L111">
<i class="fa fa-link"></i>
111
</a>
<a class="diff-line-num" data-line-number="112" href="#L112" id="L112">
<i class="fa fa-link"></i>
112
</a>
<a class="diff-line-num" data-line-number="113" href="#L113" id="L113">
<i class="fa fa-link"></i>
113
</a>
<a class="diff-line-num" data-line-number="114" href="#L114" id="L114">
<i class="fa fa-link"></i>
114
</a>
<a class="diff-line-num" data-line-number="115" href="#L115" id="L115">
<i class="fa fa-link"></i>
115
</a>
<a class="diff-line-num" data-line-number="116" href="#L116" id="L116">
<i class="fa fa-link"></i>
116
</a>
<a class="diff-line-num" data-line-number="117" href="#L117" id="L117">
<i class="fa fa-link"></i>
117
</a>
<a class="diff-line-num" data-line-number="118" href="#L118" id="L118">
<i class="fa fa-link"></i>
118
</a>
<a class="diff-line-num" data-line-number="119" href="#L119" id="L119">
<i class="fa fa-link"></i>
119
</a>
<a class="diff-line-num" data-line-number="120" href="#L120" id="L120">
<i class="fa fa-link"></i>
120
</a>
<a class="diff-line-num" data-line-number="121" href="#L121" id="L121">
<i class="fa fa-link"></i>
121
</a>
<a class="diff-line-num" data-line-number="122" href="#L122" id="L122">
<i class="fa fa-link"></i>
122
</a>
<a class="diff-line-num" data-line-number="123" href="#L123" id="L123">
<i class="fa fa-link"></i>
123
</a>
<a class="diff-line-num" data-line-number="124" href="#L124" id="L124">
<i class="fa fa-link"></i>
124
</a>
<a class="diff-line-num" data-line-number="125" href="#L125" id="L125">
<i class="fa fa-link"></i>
125
</a>
<a class="diff-line-num" data-line-number="126" href="#L126" id="L126">
<i class="fa fa-link"></i>
126
</a>
<a class="diff-line-num" data-line-number="127" href="#L127" id="L127">
<i class="fa fa-link"></i>
127
</a>
<a class="diff-line-num" data-line-number="128" href="#L128" id="L128">
<i class="fa fa-link"></i>
128
</a>
<a class="diff-line-num" data-line-number="129" href="#L129" id="L129">
<i class="fa fa-link"></i>
129
</a>
<a class="diff-line-num" data-line-number="130" href="#L130" id="L130">
<i class="fa fa-link"></i>
130
</a>
<a class="diff-line-num" data-line-number="131" href="#L131" id="L131">
<i class="fa fa-link"></i>
131
</a>
<a class="diff-line-num" data-line-number="132" href="#L132" id="L132">
<i class="fa fa-link"></i>
132
</a>
<a class="diff-line-num" data-line-number="133" href="#L133" id="L133">
<i class="fa fa-link"></i>
133
</a>
<a class="diff-line-num" data-line-number="134" href="#L134" id="L134">
<i class="fa fa-link"></i>
134
</a>
<a class="diff-line-num" data-line-number="135" href="#L135" id="L135">
<i class="fa fa-link"></i>
135
</a>
<a class="diff-line-num" data-line-number="136" href="#L136" id="L136">
<i class="fa fa-link"></i>
136
</a>
<a class="diff-line-num" data-line-number="137" href="#L137" id="L137">
<i class="fa fa-link"></i>
137
</a>
<a class="diff-line-num" data-line-number="138" href="#L138" id="L138">
<i class="fa fa-link"></i>
138
</a>
<a class="diff-line-num" data-line-number="139" href="#L139" id="L139">
<i class="fa fa-link"></i>
139
</a>
<a class="diff-line-num" data-line-number="140" href="#L140" id="L140">
<i class="fa fa-link"></i>
140
</a>
<a class="diff-line-num" data-line-number="141" href="#L141" id="L141">
<i class="fa fa-link"></i>
141
</a>
<a class="diff-line-num" data-line-number="142" href="#L142" id="L142">
<i class="fa fa-link"></i>
142
</a>
<a class="diff-line-num" data-line-number="143" href="#L143" id="L143">
<i class="fa fa-link"></i>
143
</a>
<a class="diff-line-num" data-line-number="144" href="#L144" id="L144">
<i class="fa fa-link"></i>
144
</a>
<a class="diff-line-num" data-line-number="145" href="#L145" id="L145">
<i class="fa fa-link"></i>
145
</a>
<a class="diff-line-num" data-line-number="146" href="#L146" id="L146">
<i class="fa fa-link"></i>
146
</a>
<a class="diff-line-num" data-line-number="147" href="#L147" id="L147">
<i class="fa fa-link"></i>
147
</a>
<a class="diff-line-num" data-line-number="148" href="#L148" id="L148">
<i class="fa fa-link"></i>
148
</a>
<a class="diff-line-num" data-line-number="149" href="#L149" id="L149">
<i class="fa fa-link"></i>
149
</a>
<a class="diff-line-num" data-line-number="150" href="#L150" id="L150">
<i class="fa fa-link"></i>
150
</a>
<a class="diff-line-num" data-line-number="151" href="#L151" id="L151">
<i class="fa fa-link"></i>
151
</a>
<a class="diff-line-num" data-line-number="152" href="#L152" id="L152">
<i class="fa fa-link"></i>
152
</a>
<a class="diff-line-num" data-line-number="153" href="#L153" id="L153">
<i class="fa fa-link"></i>
153
</a>
<a class="diff-line-num" data-line-number="154" href="#L154" id="L154">
<i class="fa fa-link"></i>
154
</a>
<a class="diff-line-num" data-line-number="155" href="#L155" id="L155">
<i class="fa fa-link"></i>
155
</a>
<a class="diff-line-num" data-line-number="156" href="#L156" id="L156">
<i class="fa fa-link"></i>
156
</a>
<a class="diff-line-num" data-line-number="157" href="#L157" id="L157">
<i class="fa fa-link"></i>
157
</a>
<a class="diff-line-num" data-line-number="158" href="#L158" id="L158">
<i class="fa fa-link"></i>
158
</a>
<a class="diff-line-num" data-line-number="159" href="#L159" id="L159">
<i class="fa fa-link"></i>
159
</a>
<a class="diff-line-num" data-line-number="160" href="#L160" id="L160">
<i class="fa fa-link"></i>
160
</a>
<a class="diff-line-num" data-line-number="161" href="#L161" id="L161">
<i class="fa fa-link"></i>
161
</a>
<a class="diff-line-num" data-line-number="162" href="#L162" id="L162">
<i class="fa fa-link"></i>
162
</a>
<a class="diff-line-num" data-line-number="163" href="#L163" id="L163">
<i class="fa fa-link"></i>
163
</a>
<a class="diff-line-num" data-line-number="164" href="#L164" id="L164">
<i class="fa fa-link"></i>
164
</a>
<a class="diff-line-num" data-line-number="165" href="#L165" id="L165">
<i class="fa fa-link"></i>
165
</a>
<a class="diff-line-num" data-line-number="166" href="#L166" id="L166">
<i class="fa fa-link"></i>
166
</a>
<a class="diff-line-num" data-line-number="167" href="#L167" id="L167">
<i class="fa fa-link"></i>
167
</a>
<a class="diff-line-num" data-line-number="168" href="#L168" id="L168">
<i class="fa fa-link"></i>
168
</a>
<a class="diff-line-num" data-line-number="169" href="#L169" id="L169">
<i class="fa fa-link"></i>
169
</a>
<a class="diff-line-num" data-line-number="170" href="#L170" id="L170">
<i class="fa fa-link"></i>
170
</a>
<a class="diff-line-num" data-line-number="171" href="#L171" id="L171">
<i class="fa fa-link"></i>
171
</a>
<a class="diff-line-num" data-line-number="172" href="#L172" id="L172">
<i class="fa fa-link"></i>
172
</a>
<a class="diff-line-num" data-line-number="173" href="#L173" id="L173">
<i class="fa fa-link"></i>
173
</a>
<a class="diff-line-num" data-line-number="174" href="#L174" id="L174">
<i class="fa fa-link"></i>
174
</a>
<a class="diff-line-num" data-line-number="175" href="#L175" id="L175">
<i class="fa fa-link"></i>
175
</a>
<a class="diff-line-num" data-line-number="176" href="#L176" id="L176">
<i class="fa fa-link"></i>
176
</a>
<a class="diff-line-num" data-line-number="177" href="#L177" id="L177">
<i class="fa fa-link"></i>
177
</a>
<a class="diff-line-num" data-line-number="178" href="#L178" id="L178">
<i class="fa fa-link"></i>
178
</a>
<a class="diff-line-num" data-line-number="179" href="#L179" id="L179">
<i class="fa fa-link"></i>
179
</a>
<a class="diff-line-num" data-line-number="180" href="#L180" id="L180">
<i class="fa fa-link"></i>
180
</a>
<a class="diff-line-num" data-line-number="181" href="#L181" id="L181">
<i class="fa fa-link"></i>
181
</a>
<a class="diff-line-num" data-line-number="182" href="#L182" id="L182">
<i class="fa fa-link"></i>
182
</a>
<a class="diff-line-num" data-line-number="183" href="#L183" id="L183">
<i class="fa fa-link"></i>
183
</a>
<a class="diff-line-num" data-line-number="184" href="#L184" id="L184">
<i class="fa fa-link"></i>
184
</a>
<a class="diff-line-num" data-line-number="185" href="#L185" id="L185">
<i class="fa fa-link"></i>
185
</a>
<a class="diff-line-num" data-line-number="186" href="#L186" id="L186">
<i class="fa fa-link"></i>
186
</a>
<a class="diff-line-num" data-line-number="187" href="#L187" id="L187">
<i class="fa fa-link"></i>
187
</a>
<a class="diff-line-num" data-line-number="188" href="#L188" id="L188">
<i class="fa fa-link"></i>
188
</a>
<a class="diff-line-num" data-line-number="189" href="#L189" id="L189">
<i class="fa fa-link"></i>
189
</a>
<a class="diff-line-num" data-line-number="190" href="#L190" id="L190">
<i class="fa fa-link"></i>
190
</a>
<a class="diff-line-num" data-line-number="191" href="#L191" id="L191">
<i class="fa fa-link"></i>
191
</a>
<a class="diff-line-num" data-line-number="192" href="#L192" id="L192">
<i class="fa fa-link"></i>
192
</a>
<a class="diff-line-num" data-line-number="193" href="#L193" id="L193">
<i class="fa fa-link"></i>
193
</a>
<a class="diff-line-num" data-line-number="194" href="#L194" id="L194">
<i class="fa fa-link"></i>
194
</a>
<a class="diff-line-num" data-line-number="195" href="#L195" id="L195">
<i class="fa fa-link"></i>
195
</a>
<a class="diff-line-num" data-line-number="196" href="#L196" id="L196">
<i class="fa fa-link"></i>
196
</a>
<a class="diff-line-num" data-line-number="197" href="#L197" id="L197">
<i class="fa fa-link"></i>
197
</a>
<a class="diff-line-num" data-line-number="198" href="#L198" id="L198">
<i class="fa fa-link"></i>
198
</a>
<a class="diff-line-num" data-line-number="199" href="#L199" id="L199">
<i class="fa fa-link"></i>
199
</a>
<a class="diff-line-num" data-line-number="200" href="#L200" id="L200">
<i class="fa fa-link"></i>
200
</a>
<a class="diff-line-num" data-line-number="201" href="#L201" id="L201">
<i class="fa fa-link"></i>
201
</a>
<a class="diff-line-num" data-line-number="202" href="#L202" id="L202">
<i class="fa fa-link"></i>
202
</a>
<a class="diff-line-num" data-line-number="203" href="#L203" id="L203">
<i class="fa fa-link"></i>
203
</a>
<a class="diff-line-num" data-line-number="204" href="#L204" id="L204">
<i class="fa fa-link"></i>
204
</a>
<a class="diff-line-num" data-line-number="205" href="#L205" id="L205">
<i class="fa fa-link"></i>
205
</a>
<a class="diff-line-num" data-line-number="206" href="#L206" id="L206">
<i class="fa fa-link"></i>
206
</a>
<a class="diff-line-num" data-line-number="207" href="#L207" id="L207">
<i class="fa fa-link"></i>
207
</a>
<a class="diff-line-num" data-line-number="208" href="#L208" id="L208">
<i class="fa fa-link"></i>
208
</a>
<a class="diff-line-num" data-line-number="209" href="#L209" id="L209">
<i class="fa fa-link"></i>
209
</a>
<a class="diff-line-num" data-line-number="210" href="#L210" id="L210">
<i class="fa fa-link"></i>
210
</a>
<a class="diff-line-num" data-line-number="211" href="#L211" id="L211">
<i class="fa fa-link"></i>
211
</a>
<a class="diff-line-num" data-line-number="212" href="#L212" id="L212">
<i class="fa fa-link"></i>
212
</a>
<a class="diff-line-num" data-line-number="213" href="#L213" id="L213">
<i class="fa fa-link"></i>
213
</a>
<a class="diff-line-num" data-line-number="214" href="#L214" id="L214">
<i class="fa fa-link"></i>
214
</a>
<a class="diff-line-num" data-line-number="215" href="#L215" id="L215">
<i class="fa fa-link"></i>
215
</a>
<a class="diff-line-num" data-line-number="216" href="#L216" id="L216">
<i class="fa fa-link"></i>
216
</a>
<a class="diff-line-num" data-line-number="217" href="#L217" id="L217">
<i class="fa fa-link"></i>
217
</a>
<a class="diff-line-num" data-line-number="218" href="#L218" id="L218">
<i class="fa fa-link"></i>
218
</a>
<a class="diff-line-num" data-line-number="219" href="#L219" id="L219">
<i class="fa fa-link"></i>
219
</a>
<a class="diff-line-num" data-line-number="220" href="#L220" id="L220">
<i class="fa fa-link"></i>
220
</a>
<a class="diff-line-num" data-line-number="221" href="#L221" id="L221">
<i class="fa fa-link"></i>
221
</a>
<a class="diff-line-num" data-line-number="222" href="#L222" id="L222">
<i class="fa fa-link"></i>
222
</a>
<a class="diff-line-num" data-line-number="223" href="#L223" id="L223">
<i class="fa fa-link"></i>
223
</a>
<a class="diff-line-num" data-line-number="224" href="#L224" id="L224">
<i class="fa fa-link"></i>
224
</a>
<a class="diff-line-num" data-line-number="225" href="#L225" id="L225">
<i class="fa fa-link"></i>
225
</a>
<a class="diff-line-num" data-line-number="226" href="#L226" id="L226">
<i class="fa fa-link"></i>
226
</a>
<a class="diff-line-num" data-line-number="227" href="#L227" id="L227">
<i class="fa fa-link"></i>
227
</a>
<a class="diff-line-num" data-line-number="228" href="#L228" id="L228">
<i class="fa fa-link"></i>
228
</a>
<a class="diff-line-num" data-line-number="229" href="#L229" id="L229">
<i class="fa fa-link"></i>
229
</a>
<a class="diff-line-num" data-line-number="230" href="#L230" id="L230">
<i class="fa fa-link"></i>
230
</a>
<a class="diff-line-num" data-line-number="231" href="#L231" id="L231">
<i class="fa fa-link"></i>
231
</a>
<a class="diff-line-num" data-line-number="232" href="#L232" id="L232">
<i class="fa fa-link"></i>
232
</a>
<a class="diff-line-num" data-line-number="233" href="#L233" id="L233">
<i class="fa fa-link"></i>
233
</a>
<a class="diff-line-num" data-line-number="234" href="#L234" id="L234">
<i class="fa fa-link"></i>
234
</a>
<a class="diff-line-num" data-line-number="235" href="#L235" id="L235">
<i class="fa fa-link"></i>
235
</a>
<a class="diff-line-num" data-line-number="236" href="#L236" id="L236">
<i class="fa fa-link"></i>
236
</a>
<a class="diff-line-num" data-line-number="237" href="#L237" id="L237">
<i class="fa fa-link"></i>
237
</a>
<a class="diff-line-num" data-line-number="238" href="#L238" id="L238">
<i class="fa fa-link"></i>
238
</a>
<a class="diff-line-num" data-line-number="239" href="#L239" id="L239">
<i class="fa fa-link"></i>
239
</a>
<a class="diff-line-num" data-line-number="240" href="#L240" id="L240">
<i class="fa fa-link"></i>
240
</a>
<a class="diff-line-num" data-line-number="241" href="#L241" id="L241">
<i class="fa fa-link"></i>
241
</a>
<a class="diff-line-num" data-line-number="242" href="#L242" id="L242">
<i class="fa fa-link"></i>
242
</a>
<a class="diff-line-num" data-line-number="243" href="#L243" id="L243">
<i class="fa fa-link"></i>
243
</a>
<a class="diff-line-num" data-line-number="244" href="#L244" id="L244">
<i class="fa fa-link"></i>
244
</a>
<a class="diff-line-num" data-line-number="245" href="#L245" id="L245">
<i class="fa fa-link"></i>
245
</a>
<a class="diff-line-num" data-line-number="246" href="#L246" id="L246">
<i class="fa fa-link"></i>
246
</a>
<a class="diff-line-num" data-line-number="247" href="#L247" id="L247">
<i class="fa fa-link"></i>
247
</a>
<a class="diff-line-num" data-line-number="248" href="#L248" id="L248">
<i class="fa fa-link"></i>
248
</a>
<a class="diff-line-num" data-line-number="249" href="#L249" id="L249">
<i class="fa fa-link"></i>
249
</a>
<a class="diff-line-num" data-line-number="250" href="#L250" id="L250">
<i class="fa fa-link"></i>
250
</a>
<a class="diff-line-num" data-line-number="251" href="#L251" id="L251">
<i class="fa fa-link"></i>
251
</a>
<a class="diff-line-num" data-line-number="252" href="#L252" id="L252">
<i class="fa fa-link"></i>
252
</a>
<a class="diff-line-num" data-line-number="253" href="#L253" id="L253">
<i class="fa fa-link"></i>
253
</a>
<a class="diff-line-num" data-line-number="254" href="#L254" id="L254">
<i class="fa fa-link"></i>
254
</a>
<a class="diff-line-num" data-line-number="255" href="#L255" id="L255">
<i class="fa fa-link"></i>
255
</a>
<a class="diff-line-num" data-line-number="256" href="#L256" id="L256">
<i class="fa fa-link"></i>
256
</a>
<a class="diff-line-num" data-line-number="257" href="#L257" id="L257">
<i class="fa fa-link"></i>
257
</a>
<a class="diff-line-num" data-line-number="258" href="#L258" id="L258">
<i class="fa fa-link"></i>
258
</a>
<a class="diff-line-num" data-line-number="259" href="#L259" id="L259">
<i class="fa fa-link"></i>
259
</a>
<a class="diff-line-num" data-line-number="260" href="#L260" id="L260">
<i class="fa fa-link"></i>
260
</a>
<a class="diff-line-num" data-line-number="261" href="#L261" id="L261">
<i class="fa fa-link"></i>
261
</a>
<a class="diff-line-num" data-line-number="262" href="#L262" id="L262">
<i class="fa fa-link"></i>
262
</a>
<a class="diff-line-num" data-line-number="263" href="#L263" id="L263">
<i class="fa fa-link"></i>
263
</a>
<a class="diff-line-num" data-line-number="264" href="#L264" id="L264">
<i class="fa fa-link"></i>
264
</a>
<a class="diff-line-num" data-line-number="265" href="#L265" id="L265">
<i class="fa fa-link"></i>
265
</a>
<a class="diff-line-num" data-line-number="266" href="#L266" id="L266">
<i class="fa fa-link"></i>
266
</a>
<a class="diff-line-num" data-line-number="267" href="#L267" id="L267">
<i class="fa fa-link"></i>
267
</a>
<a class="diff-line-num" data-line-number="268" href="#L268" id="L268">
<i class="fa fa-link"></i>
268
</a>
<a class="diff-line-num" data-line-number="269" href="#L269" id="L269">
<i class="fa fa-link"></i>
269
</a>
<a class="diff-line-num" data-line-number="270" href="#L270" id="L270">
<i class="fa fa-link"></i>
270
</a>
<a class="diff-line-num" data-line-number="271" href="#L271" id="L271">
<i class="fa fa-link"></i>
271
</a>
<a class="diff-line-num" data-line-number="272" href="#L272" id="L272">
<i class="fa fa-link"></i>
272
</a>
<a class="diff-line-num" data-line-number="273" href="#L273" id="L273">
<i class="fa fa-link"></i>
273
</a>
<a class="diff-line-num" data-line-number="274" href="#L274" id="L274">
<i class="fa fa-link"></i>
274
</a>
<a class="diff-line-num" data-line-number="275" href="#L275" id="L275">
<i class="fa fa-link"></i>
275
</a>
<a class="diff-line-num" data-line-number="276" href="#L276" id="L276">
<i class="fa fa-link"></i>
276
</a>
<a class="diff-line-num" data-line-number="277" href="#L277" id="L277">
<i class="fa fa-link"></i>
277
</a>
<a class="diff-line-num" data-line-number="278" href="#L278" id="L278">
<i class="fa fa-link"></i>
278
</a>
<a class="diff-line-num" data-line-number="279" href="#L279" id="L279">
<i class="fa fa-link"></i>
279
</a>
<a class="diff-line-num" data-line-number="280" href="#L280" id="L280">
<i class="fa fa-link"></i>
280
</a>
<a class="diff-line-num" data-line-number="281" href="#L281" id="L281">
<i class="fa fa-link"></i>
281
</a>
<a class="diff-line-num" data-line-number="282" href="#L282" id="L282">
<i class="fa fa-link"></i>
282
</a>
<a class="diff-line-num" data-line-number="283" href="#L283" id="L283">
<i class="fa fa-link"></i>
283
</a>
<a class="diff-line-num" data-line-number="284" href="#L284" id="L284">
<i class="fa fa-link"></i>
284
</a>
<a class="diff-line-num" data-line-number="285" href="#L285" id="L285">
<i class="fa fa-link"></i>
285
</a>
<a class="diff-line-num" data-line-number="286" href="#L286" id="L286">
<i class="fa fa-link"></i>
286
</a>
<a class="diff-line-num" data-line-number="287" href="#L287" id="L287">
<i class="fa fa-link"></i>
287
</a>
<a class="diff-line-num" data-line-number="288" href="#L288" id="L288">
<i class="fa fa-link"></i>
288
</a>
<a class="diff-line-num" data-line-number="289" href="#L289" id="L289">
<i class="fa fa-link"></i>
289
</a>
<a class="diff-line-num" data-line-number="290" href="#L290" id="L290">
<i class="fa fa-link"></i>
290
</a>
<a class="diff-line-num" data-line-number="291" href="#L291" id="L291">
<i class="fa fa-link"></i>
291
</a>
<a class="diff-line-num" data-line-number="292" href="#L292" id="L292">
<i class="fa fa-link"></i>
292
</a>
<a class="diff-line-num" data-line-number="293" href="#L293" id="L293">
<i class="fa fa-link"></i>
293
</a>
<a class="diff-line-num" data-line-number="294" href="#L294" id="L294">
<i class="fa fa-link"></i>
294
</a>
<a class="diff-line-num" data-line-number="295" href="#L295" id="L295">
<i class="fa fa-link"></i>
295
</a>
<a class="diff-line-num" data-line-number="296" href="#L296" id="L296">
<i class="fa fa-link"></i>
296
</a>
<a class="diff-line-num" data-line-number="297" href="#L297" id="L297">
<i class="fa fa-link"></i>
297
</a>
<a class="diff-line-num" data-line-number="298" href="#L298" id="L298">
<i class="fa fa-link"></i>
298
</a>
<a class="diff-line-num" data-line-number="299" href="#L299" id="L299">
<i class="fa fa-link"></i>
299
</a>
<a class="diff-line-num" data-line-number="300" href="#L300" id="L300">
<i class="fa fa-link"></i>
300
</a>
<a class="diff-line-num" data-line-number="301" href="#L301" id="L301">
<i class="fa fa-link"></i>
301
</a>
<a class="diff-line-num" data-line-number="302" href="#L302" id="L302">
<i class="fa fa-link"></i>
302
</a>
<a class="diff-line-num" data-line-number="303" href="#L303" id="L303">
<i class="fa fa-link"></i>
303
</a>
<a class="diff-line-num" data-line-number="304" href="#L304" id="L304">
<i class="fa fa-link"></i>
304
</a>
<a class="diff-line-num" data-line-number="305" href="#L305" id="L305">
<i class="fa fa-link"></i>
305
</a>
<a class="diff-line-num" data-line-number="306" href="#L306" id="L306">
<i class="fa fa-link"></i>
306
</a>
<a class="diff-line-num" data-line-number="307" href="#L307" id="L307">
<i class="fa fa-link"></i>
307
</a>
<a class="diff-line-num" data-line-number="308" href="#L308" id="L308">
<i class="fa fa-link"></i>
308
</a>
<a class="diff-line-num" data-line-number="309" href="#L309" id="L309">
<i class="fa fa-link"></i>
309
</a>
<a class="diff-line-num" data-line-number="310" href="#L310" id="L310">
<i class="fa fa-link"></i>
310
</a>
<a class="diff-line-num" data-line-number="311" href="#L311" id="L311">
<i class="fa fa-link"></i>
311
</a>
<a class="diff-line-num" data-line-number="312" href="#L312" id="L312">
<i class="fa fa-link"></i>
312
</a>
<a class="diff-line-num" data-line-number="313" href="#L313" id="L313">
<i class="fa fa-link"></i>
313
</a>
<a class="diff-line-num" data-line-number="314" href="#L314" id="L314">
<i class="fa fa-link"></i>
314
</a>
<a class="diff-line-num" data-line-number="315" href="#L315" id="L315">
<i class="fa fa-link"></i>
315
</a>
<a class="diff-line-num" data-line-number="316" href="#L316" id="L316">
<i class="fa fa-link"></i>
316
</a>
<a class="diff-line-num" data-line-number="317" href="#L317" id="L317">
<i class="fa fa-link"></i>
317
</a>
<a class="diff-line-num" data-line-number="318" href="#L318" id="L318">
<i class="fa fa-link"></i>
318
</a>
<a class="diff-line-num" data-line-number="319" href="#L319" id="L319">
<i class="fa fa-link"></i>
319
</a>
<a class="diff-line-num" data-line-number="320" href="#L320" id="L320">
<i class="fa fa-link"></i>
320
</a>
<a class="diff-line-num" data-line-number="321" href="#L321" id="L321">
<i class="fa fa-link"></i>
321
</a>
<a class="diff-line-num" data-line-number="322" href="#L322" id="L322">
<i class="fa fa-link"></i>
322
</a>
<a class="diff-line-num" data-line-number="323" href="#L323" id="L323">
<i class="fa fa-link"></i>
323
</a>
<a class="diff-line-num" data-line-number="324" href="#L324" id="L324">
<i class="fa fa-link"></i>
324
</a>
<a class="diff-line-num" data-line-number="325" href="#L325" id="L325">
<i class="fa fa-link"></i>
325
</a>
<a class="diff-line-num" data-line-number="326" href="#L326" id="L326">
<i class="fa fa-link"></i>
326
</a>
<a class="diff-line-num" data-line-number="327" href="#L327" id="L327">
<i class="fa fa-link"></i>
327
</a>
<a class="diff-line-num" data-line-number="328" href="#L328" id="L328">
<i class="fa fa-link"></i>
328
</a>
<a class="diff-line-num" data-line-number="329" href="#L329" id="L329">
<i class="fa fa-link"></i>
329
</a>
<a class="diff-line-num" data-line-number="330" href="#L330" id="L330">
<i class="fa fa-link"></i>
330
</a>
<a class="diff-line-num" data-line-number="331" href="#L331" id="L331">
<i class="fa fa-link"></i>
331
</a>
<a class="diff-line-num" data-line-number="332" href="#L332" id="L332">
<i class="fa fa-link"></i>
332
</a>
<a class="diff-line-num" data-line-number="333" href="#L333" id="L333">
<i class="fa fa-link"></i>
333
</a>
<a class="diff-line-num" data-line-number="334" href="#L334" id="L334">
<i class="fa fa-link"></i>
334
</a>
<a class="diff-line-num" data-line-number="335" href="#L335" id="L335">
<i class="fa fa-link"></i>
335
</a>
<a class="diff-line-num" data-line-number="336" href="#L336" id="L336">
<i class="fa fa-link"></i>
336
</a>
<a class="diff-line-num" data-line-number="337" href="#L337" id="L337">
<i class="fa fa-link"></i>
337
</a>
<a class="diff-line-num" data-line-number="338" href="#L338" id="L338">
<i class="fa fa-link"></i>
338
</a>
<a class="diff-line-num" data-line-number="339" href="#L339" id="L339">
<i class="fa fa-link"></i>
339
</a>
<a class="diff-line-num" data-line-number="340" href="#L340" id="L340">
<i class="fa fa-link"></i>
340
</a>
<a class="diff-line-num" data-line-number="341" href="#L341" id="L341">
<i class="fa fa-link"></i>
341
</a>
<a class="diff-line-num" data-line-number="342" href="#L342" id="L342">
<i class="fa fa-link"></i>
342
</a>
<a class="diff-line-num" data-line-number="343" href="#L343" id="L343">
<i class="fa fa-link"></i>
343
</a>
<a class="diff-line-num" data-line-number="344" href="#L344" id="L344">
<i class="fa fa-link"></i>
344
</a>
<a class="diff-line-num" data-line-number="345" href="#L345" id="L345">
<i class="fa fa-link"></i>
345
</a>
<a class="diff-line-num" data-line-number="346" href="#L346" id="L346">
<i class="fa fa-link"></i>
346
</a>
<a class="diff-line-num" data-line-number="347" href="#L347" id="L347">
<i class="fa fa-link"></i>
347
</a>
<a class="diff-line-num" data-line-number="348" href="#L348" id="L348">
<i class="fa fa-link"></i>
348
</a>
<a class="diff-line-num" data-line-number="349" href="#L349" id="L349">
<i class="fa fa-link"></i>
349
</a>
<a class="diff-line-num" data-line-number="350" href="#L350" id="L350">
<i class="fa fa-link"></i>
350
</a>
<a class="diff-line-num" data-line-number="351" href="#L351" id="L351">
<i class="fa fa-link"></i>
351
</a>
<a class="diff-line-num" data-line-number="352" href="#L352" id="L352">
<i class="fa fa-link"></i>
352
</a>
<a class="diff-line-num" data-line-number="353" href="#L353" id="L353">
<i class="fa fa-link"></i>
353
</a>
<a class="diff-line-num" data-line-number="354" href="#L354" id="L354">
<i class="fa fa-link"></i>
354
</a>
<a class="diff-line-num" data-line-number="355" href="#L355" id="L355">
<i class="fa fa-link"></i>
355
</a>
<a class="diff-line-num" data-line-number="356" href="#L356" id="L356">
<i class="fa fa-link"></i>
356
</a>
<a class="diff-line-num" data-line-number="357" href="#L357" id="L357">
<i class="fa fa-link"></i>
357
</a>
<a class="diff-line-num" data-line-number="358" href="#L358" id="L358">
<i class="fa fa-link"></i>
358
</a>
<a class="diff-line-num" data-line-number="359" href="#L359" id="L359">
<i class="fa fa-link"></i>
359
</a>
<a class="diff-line-num" data-line-number="360" href="#L360" id="L360">
<i class="fa fa-link"></i>
360
</a>
<a class="diff-line-num" data-line-number="361" href="#L361" id="L361">
<i class="fa fa-link"></i>
361
</a>
<a class="diff-line-num" data-line-number="362" href="#L362" id="L362">
<i class="fa fa-link"></i>
362
</a>
<a class="diff-line-num" data-line-number="363" href="#L363" id="L363">
<i class="fa fa-link"></i>
363
</a>
<a class="diff-line-num" data-line-number="364" href="#L364" id="L364">
<i class="fa fa-link"></i>
364
</a>
<a class="diff-line-num" data-line-number="365" href="#L365" id="L365">
<i class="fa fa-link"></i>
365
</a>
<a class="diff-line-num" data-line-number="366" href="#L366" id="L366">
<i class="fa fa-link"></i>
366
</a>
<a class="diff-line-num" data-line-number="367" href="#L367" id="L367">
<i class="fa fa-link"></i>
367
</a>
<a class="diff-line-num" data-line-number="368" href="#L368" id="L368">
<i class="fa fa-link"></i>
368
</a>
<a class="diff-line-num" data-line-number="369" href="#L369" id="L369">
<i class="fa fa-link"></i>
369
</a>
<a class="diff-line-num" data-line-number="370" href="#L370" id="L370">
<i class="fa fa-link"></i>
370
</a>
<a class="diff-line-num" data-line-number="371" href="#L371" id="L371">
<i class="fa fa-link"></i>
371
</a>
<a class="diff-line-num" data-line-number="372" href="#L372" id="L372">
<i class="fa fa-link"></i>
372
</a>
<a class="diff-line-num" data-line-number="373" href="#L373" id="L373">
<i class="fa fa-link"></i>
373
</a>
<a class="diff-line-num" data-line-number="374" href="#L374" id="L374">
<i class="fa fa-link"></i>
374
</a>
<a class="diff-line-num" data-line-number="375" href="#L375" id="L375">
<i class="fa fa-link"></i>
375
</a>
<a class="diff-line-num" data-line-number="376" href="#L376" id="L376">
<i class="fa fa-link"></i>
376
</a>
<a class="diff-line-num" data-line-number="377" href="#L377" id="L377">
<i class="fa fa-link"></i>
377
</a>
<a class="diff-line-num" data-line-number="378" href="#L378" id="L378">
<i class="fa fa-link"></i>
378
</a>
<a class="diff-line-num" data-line-number="379" href="#L379" id="L379">
<i class="fa fa-link"></i>
379
</a>
<a class="diff-line-num" data-line-number="380" href="#L380" id="L380">
<i class="fa fa-link"></i>
380
</a>
<a class="diff-line-num" data-line-number="381" href="#L381" id="L381">
<i class="fa fa-link"></i>
381
</a>
<a class="diff-line-num" data-line-number="382" href="#L382" id="L382">
<i class="fa fa-link"></i>
382
</a>
<a class="diff-line-num" data-line-number="383" href="#L383" id="L383">
<i class="fa fa-link"></i>
383
</a>
<a class="diff-line-num" data-line-number="384" href="#L384" id="L384">
<i class="fa fa-link"></i>
384
</a>
<a class="diff-line-num" data-line-number="385" href="#L385" id="L385">
<i class="fa fa-link"></i>
385
</a>
<a class="diff-line-num" data-line-number="386" href="#L386" id="L386">
<i class="fa fa-link"></i>
386
</a>
<a class="diff-line-num" data-line-number="387" href="#L387" id="L387">
<i class="fa fa-link"></i>
387
</a>
<a class="diff-line-num" data-line-number="388" href="#L388" id="L388">
<i class="fa fa-link"></i>
388
</a>
<a class="diff-line-num" data-line-number="389" href="#L389" id="L389">
<i class="fa fa-link"></i>
389
</a>
<a class="diff-line-num" data-line-number="390" href="#L390" id="L390">
<i class="fa fa-link"></i>
390
</a>
<a class="diff-line-num" data-line-number="391" href="#L391" id="L391">
<i class="fa fa-link"></i>
391
</a>
<a class="diff-line-num" data-line-number="392" href="#L392" id="L392">
<i class="fa fa-link"></i>
392
</a>
<a class="diff-line-num" data-line-number="393" href="#L393" id="L393">
<i class="fa fa-link"></i>
393
</a>
<a class="diff-line-num" data-line-number="394" href="#L394" id="L394">
<i class="fa fa-link"></i>
394
</a>
<a class="diff-line-num" data-line-number="395" href="#L395" id="L395">
<i class="fa fa-link"></i>
395
</a>
<a class="diff-line-num" data-line-number="396" href="#L396" id="L396">
<i class="fa fa-link"></i>
396
</a>
<a class="diff-line-num" data-line-number="397" href="#L397" id="L397">
<i class="fa fa-link"></i>
397
</a>
<a class="diff-line-num" data-line-number="398" href="#L398" id="L398">
<i class="fa fa-link"></i>
398
</a>
<a class="diff-line-num" data-line-number="399" href="#L399" id="L399">
<i class="fa fa-link"></i>
399
</a>
<a class="diff-line-num" data-line-number="400" href="#L400" id="L400">
<i class="fa fa-link"></i>
400
</a>
<a class="diff-line-num" data-line-number="401" href="#L401" id="L401">
<i class="fa fa-link"></i>
401
</a>
<a class="diff-line-num" data-line-number="402" href="#L402" id="L402">
<i class="fa fa-link"></i>
402
</a>
<a class="diff-line-num" data-line-number="403" href="#L403" id="L403">
<i class="fa fa-link"></i>
403
</a>
<a class="diff-line-num" data-line-number="404" href="#L404" id="L404">
<i class="fa fa-link"></i>
404
</a>
<a class="diff-line-num" data-line-number="405" href="#L405" id="L405">
<i class="fa fa-link"></i>
405
</a>
<a class="diff-line-num" data-line-number="406" href="#L406" id="L406">
<i class="fa fa-link"></i>
406
</a>
<a class="diff-line-num" data-line-number="407" href="#L407" id="L407">
<i class="fa fa-link"></i>
407
</a>
<a class="diff-line-num" data-line-number="408" href="#L408" id="L408">
<i class="fa fa-link"></i>
408
</a>
<a class="diff-line-num" data-line-number="409" href="#L409" id="L409">
<i class="fa fa-link"></i>
409
</a>
<a class="diff-line-num" data-line-number="410" href="#L410" id="L410">
<i class="fa fa-link"></i>
410
</a>
<a class="diff-line-num" data-line-number="411" href="#L411" id="L411">
<i class="fa fa-link"></i>
411
</a>
<a class="diff-line-num" data-line-number="412" href="#L412" id="L412">
<i class="fa fa-link"></i>
412
</a>
<a class="diff-line-num" data-line-number="413" href="#L413" id="L413">
<i class="fa fa-link"></i>
413
</a>
<a class="diff-line-num" data-line-number="414" href="#L414" id="L414">
<i class="fa fa-link"></i>
414
</a>
<a class="diff-line-num" data-line-number="415" href="#L415" id="L415">
<i class="fa fa-link"></i>
415
</a>
<a class="diff-line-num" data-line-number="416" href="#L416" id="L416">
<i class="fa fa-link"></i>
416
</a>
<a class="diff-line-num" data-line-number="417" href="#L417" id="L417">
<i class="fa fa-link"></i>
417
</a>
<a class="diff-line-num" data-line-number="418" href="#L418" id="L418">
<i class="fa fa-link"></i>
418
</a>
<a class="diff-line-num" data-line-number="419" href="#L419" id="L419">
<i class="fa fa-link"></i>
419
</a>
<a class="diff-line-num" data-line-number="420" href="#L420" id="L420">
<i class="fa fa-link"></i>
420
</a>
<a class="diff-line-num" data-line-number="421" href="#L421" id="L421">
<i class="fa fa-link"></i>
421
</a>
<a class="diff-line-num" data-line-number="422" href="#L422" id="L422">
<i class="fa fa-link"></i>
422
</a>
<a class="diff-line-num" data-line-number="423" href="#L423" id="L423">
<i class="fa fa-link"></i>
423
</a>
<a class="diff-line-num" data-line-number="424" href="#L424" id="L424">
<i class="fa fa-link"></i>
424
</a>
<a class="diff-line-num" data-line-number="425" href="#L425" id="L425">
<i class="fa fa-link"></i>
425
</a>
<a class="diff-line-num" data-line-number="426" href="#L426" id="L426">
<i class="fa fa-link"></i>
426
</a>
<a class="diff-line-num" data-line-number="427" href="#L427" id="L427">
<i class="fa fa-link"></i>
427
</a>
<a class="diff-line-num" data-line-number="428" href="#L428" id="L428">
<i class="fa fa-link"></i>
428
</a>
<a class="diff-line-num" data-line-number="429" href="#L429" id="L429">
<i class="fa fa-link"></i>
429
</a>
<a class="diff-line-num" data-line-number="430" href="#L430" id="L430">
<i class="fa fa-link"></i>
430
</a>
<a class="diff-line-num" data-line-number="431" href="#L431" id="L431">
<i class="fa fa-link"></i>
431
</a>
<a class="diff-line-num" data-line-number="432" href="#L432" id="L432">
<i class="fa fa-link"></i>
432
</a>
<a class="diff-line-num" data-line-number="433" href="#L433" id="L433">
<i class="fa fa-link"></i>
433
</a>
<a class="diff-line-num" data-line-number="434" href="#L434" id="L434">
<i class="fa fa-link"></i>
434
</a>
<a class="diff-line-num" data-line-number="435" href="#L435" id="L435">
<i class="fa fa-link"></i>
435
</a>
<a class="diff-line-num" data-line-number="436" href="#L436" id="L436">
<i class="fa fa-link"></i>
436
</a>
<a class="diff-line-num" data-line-number="437" href="#L437" id="L437">
<i class="fa fa-link"></i>
437
</a>
<a class="diff-line-num" data-line-number="438" href="#L438" id="L438">
<i class="fa fa-link"></i>
438
</a>
<a class="diff-line-num" data-line-number="439" href="#L439" id="L439">
<i class="fa fa-link"></i>
439
</a>
<a class="diff-line-num" data-line-number="440" href="#L440" id="L440">
<i class="fa fa-link"></i>
440
</a>
<a class="diff-line-num" data-line-number="441" href="#L441" id="L441">
<i class="fa fa-link"></i>
441
</a>
<a class="diff-line-num" data-line-number="442" href="#L442" id="L442">
<i class="fa fa-link"></i>
442
</a>
<a class="diff-line-num" data-line-number="443" href="#L443" id="L443">
<i class="fa fa-link"></i>
443
</a>
<a class="diff-line-num" data-line-number="444" href="#L444" id="L444">
<i class="fa fa-link"></i>
444
</a>
<a class="diff-line-num" data-line-number="445" href="#L445" id="L445">
<i class="fa fa-link"></i>
445
</a>
<a class="diff-line-num" data-line-number="446" href="#L446" id="L446">
<i class="fa fa-link"></i>
446
</a>
<a class="diff-line-num" data-line-number="447" href="#L447" id="L447">
<i class="fa fa-link"></i>
447
</a>
<a class="diff-line-num" data-line-number="448" href="#L448" id="L448">
<i class="fa fa-link"></i>
448
</a>
<a class="diff-line-num" data-line-number="449" href="#L449" id="L449">
<i class="fa fa-link"></i>
449
</a>
</div>
<div class="blob-content" data-blob-id="a72704f4bcfa3fde56536fdcc2ef2702660e3c6f">
<pre class="code highlight"><code><span id="LC1" class="line"><span class="c1">//</span></span>
<span id="LC2" class="line"><span class="c1">// Book:      OpenCL(R) Programming Guide</span></span>
<span id="LC3" class="line"><span class="c1">// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg</span></span>
<span id="LC4" class="line"><span class="c1">// ISBN-10:   0-321-74964-2</span></span>
<span id="LC5" class="line"><span class="c1">// ISBN-13:   978-0-321-74964-2</span></span>
<span id="LC6" class="line"><span class="c1">// Publisher: Addison-Wesley Professional</span></span>
<span id="LC7" class="line"><span class="c1">// URLs:      http://safari.informit.com/9780132488006/</span></span>
<span id="LC8" class="line"><span class="c1">//            http://www.openclprogrammingguide.com</span></span>
<span id="LC9" class="line"><span class="c1">//</span></span>
<span id="LC10" class="line"></span>
<span id="LC11" class="line"><span class="c1">// ImageFilter2D.cpp</span></span>
<span id="LC12" class="line"><span class="c1">//</span></span>
<span id="LC13" class="line"><span class="c1">//    This example demonstrates performing gaussian filtering on a 2D image using</span></span>
<span id="LC14" class="line"><span class="c1">//    OpenCL</span></span>
<span id="LC15" class="line"><span class="c1">//</span></span>
<span id="LC16" class="line"><span class="c1">//    Requires FreeImage library for image I/O:</span></span>
<span id="LC17" class="line"><span class="c1">//      http://freeimage.sourceforge.net/</span></span>
<span id="LC18" class="line"></span>
<span id="LC19" class="line"><span class="cp">#include &lt;iostream&gt;</span></span>
<span id="LC20" class="line"><span class="cp">#include &lt;fstream&gt;</span></span>
<span id="LC21" class="line"><span class="cp">#include &lt;sstream&gt;</span></span>
<span id="LC22" class="line"><span class="cp">#include &lt;string.h&gt;</span></span>
<span id="LC23" class="line"></span>
<span id="LC24" class="line"><span class="cp">#ifdef __APPLE__</span></span>
<span id="LC25" class="line"><span class="cp">#include &lt;OpenCL/cl.h&gt;</span></span>
<span id="LC26" class="line"><span class="cp">#else</span></span>
<span id="LC27" class="line"><span class="cp">#include &lt;CL/cl.h&gt;</span></span>
<span id="LC28" class="line"><span class="cp">#endif</span></span>
<span id="LC29" class="line"></span>
<span id="LC30" class="line"><span class="cp">#include "FreeImage.h"</span></span>
<span id="LC31" class="line"></span>
<span id="LC32" class="line"><span class="c1">///</span></span>
<span id="LC33" class="line"><span class="c1">//  Create an OpenCL context on the first available platform using</span></span>
<span id="LC34" class="line"><span class="c1">//  either a GPU or CPU depending on what is available.</span></span>
<span id="LC35" class="line"><span class="c1">//</span></span>
<span id="LC36" class="line"><span class="n">cl_context</span> <span class="nf">CreateContext</span><span class="p">()</span></span>
<span id="LC37" class="line"><span class="p">{</span></span>
<span id="LC38" class="line">    <span class="n">cl_int</span> <span class="n">errNum</span><span class="p">;</span></span>
<span id="LC39" class="line">    <span class="n">cl_uint</span> <span class="n">numPlatforms</span><span class="p">;</span></span>
<span id="LC40" class="line">    <span class="n">cl_platform_id</span> <span class="n">firstPlatformId</span><span class="p">;</span></span>
<span id="LC41" class="line">    <span class="n">cl_context</span> <span class="n">context</span> <span class="o">=</span> <span class="nb">NULL</span><span class="p">;</span></span>
<span id="LC42" class="line"></span>
<span id="LC43" class="line">    <span class="c1">// First, select an OpenCL platform to run on.  For this example, we</span></span>
<span id="LC44" class="line">    <span class="c1">// simply choose the first available platform.  Normally, you would</span></span>
<span id="LC45" class="line">    <span class="c1">// query for all available platforms and select the most appropriate one.</span></span>
<span id="LC46" class="line">    <span class="n">errNum</span> <span class="o">=</span> <span class="n">clGetPlatformIDs</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="o">&amp;</span><span class="n">firstPlatformId</span><span class="p">,</span> <span class="o">&amp;</span><span class="n">numPlatforms</span><span class="p">);</span></span>
<span id="LC47" class="line">    <span class="k">if</span> <span class="p">(</span><span class="n">errNum</span> <span class="o">!=</span> <span class="n">CL_SUCCESS</span> <span class="o">||</span> <span class="n">numPlatforms</span> <span class="o">&lt;=</span> <span class="mi">0</span><span class="p">)</span></span>
<span id="LC48" class="line">    <span class="p">{</span></span>
<span id="LC49" class="line">        <span class="n">std</span><span class="o">::</span><span class="n">cerr</span> <span class="o">&lt;&lt;</span> <span class="s">"Failed to find any OpenCL platforms."</span> <span class="o">&lt;&lt;</span> <span class="n">std</span><span class="o">::</span><span class="n">endl</span><span class="p">;</span></span>
<span id="LC50" class="line">        <span class="k">return</span> <span class="nb">NULL</span><span class="p">;</span></span>
<span id="LC51" class="line">    <span class="p">}</span></span>
<span id="LC52" class="line"></span>
<span id="LC53" class="line">    <span class="c1">// Next, create an OpenCL context on the platform.  Attempt to</span></span>
<span id="LC54" class="line">    <span class="c1">// create a GPU-based context, and if that fails, try to create</span></span>
<span id="LC55" class="line">    <span class="c1">// a CPU-based context.</span></span>
<span id="LC56" class="line">    <span class="n">cl_context_properties</span> <span class="n">contextProperties</span><span class="p">[]</span> <span class="o">=</span></span>
<span id="LC57" class="line">    <span class="p">{</span></span>
<span id="LC58" class="line">        <span class="n">CL_CONTEXT_PLATFORM</span><span class="p">,</span></span>
<span id="LC59" class="line">        <span class="p">(</span><span class="n">cl_context_properties</span><span class="p">)</span><span class="n">firstPlatformId</span><span class="p">,</span></span>
<span id="LC60" class="line">        <span class="mi">0</span></span>
<span id="LC61" class="line">    <span class="p">};</span></span>
<span id="LC62" class="line">    <span class="n">context</span> <span class="o">=</span> <span class="n">clCreateContextFromType</span><span class="p">(</span><span class="n">contextProperties</span><span class="p">,</span> <span class="n">CL_DEVICE_TYPE_GPU</span><span class="p">,</span></span>
<span id="LC63" class="line">                                      <span class="nb">NULL</span><span class="p">,</span> <span class="nb">NULL</span><span class="p">,</span> <span class="o">&amp;</span><span class="n">errNum</span><span class="p">);</span></span>
<span id="LC64" class="line">    <span class="k">if</span> <span class="p">(</span><span class="n">errNum</span> <span class="o">!=</span> <span class="n">CL_SUCCESS</span><span class="p">)</span></span>
<span id="LC65" class="line">    <span class="p">{</span></span>
<span id="LC66" class="line">        <span class="n">std</span><span class="o">::</span><span class="n">cout</span> <span class="o">&lt;&lt;</span> <span class="s">"Could not create GPU context, trying CPU..."</span> <span class="o">&lt;&lt;</span> <span class="n">std</span><span class="o">::</span><span class="n">endl</span><span class="p">;</span></span>
<span id="LC67" class="line">        <span class="n">context</span> <span class="o">=</span> <span class="n">clCreateContextFromType</span><span class="p">(</span><span class="n">contextProperties</span><span class="p">,</span> <span class="n">CL_DEVICE_TYPE_CPU</span><span class="p">,</span></span>
<span id="LC68" class="line">                                          <span class="nb">NULL</span><span class="p">,</span> <span class="nb">NULL</span><span class="p">,</span> <span class="o">&amp;</span><span class="n">errNum</span><span class="p">);</span></span>
<span id="LC69" class="line">        <span class="k">if</span> <span class="p">(</span><span class="n">errNum</span> <span class="o">!=</span> <span class="n">CL_SUCCESS</span><span class="p">)</span></span>
<span id="LC70" class="line">        <span class="p">{</span></span>
<span id="LC71" class="line">            <span class="n">std</span><span class="o">::</span><span class="n">cerr</span> <span class="o">&lt;&lt;</span> <span class="s">"Failed to create an OpenCL GPU or CPU context."</span> <span class="o">&lt;&lt;</span> <span class="n">std</span><span class="o">::</span><span class="n">endl</span><span class="p">;</span></span>
<span id="LC72" class="line">            <span class="k">return</span> <span class="nb">NULL</span><span class="p">;</span></span>
<span id="LC73" class="line">        <span class="p">}</span></span>
<span id="LC74" class="line">    <span class="p">}</span></span>
<span id="LC75" class="line"></span>
<span id="LC76" class="line">    <span class="k">return</span> <span class="n">context</span><span class="p">;</span></span>
<span id="LC77" class="line"><span class="p">}</span></span>
<span id="LC78" class="line"></span>
<span id="LC79" class="line"><span class="c1">///</span></span>
<span id="LC80" class="line"><span class="c1">//  Create a command queue on the first device available on the</span></span>
<span id="LC81" class="line"><span class="c1">//  context</span></span>
<span id="LC82" class="line"><span class="c1">//</span></span>
<span id="LC83" class="line"><span class="n">cl_command_queue</span> <span class="nf">CreateCommandQueue</span><span class="p">(</span><span class="n">cl_context</span> <span class="n">context</span><span class="p">,</span> <span class="n">cl_device_id</span> <span class="o">*</span><span class="n">device</span><span class="p">)</span></span>
<span id="LC84" class="line"><span class="p">{</span></span>
<span id="LC85" class="line">    <span class="n">cl_int</span> <span class="n">errNum</span><span class="p">;</span></span>
<span id="LC86" class="line">    <span class="n">cl_device_id</span> <span class="o">*</span><span class="n">devices</span><span class="p">;</span></span>
<span id="LC87" class="line">    <span class="n">cl_command_queue</span> <span class="n">commandQueue</span> <span class="o">=</span> <span class="nb">NULL</span><span class="p">;</span></span>
<span id="LC88" class="line">    <span class="kt">size_t</span> <span class="n">deviceBufferSize</span> <span class="o">=</span> <span class="o">-</span><span class="mi">1</span><span class="p">;</span></span>
<span id="LC89" class="line"></span>
<span id="LC90" class="line">    <span class="c1">// First get the size of the devices buffer</span></span>
<span id="LC91" class="line">    <span class="n">errNum</span> <span class="o">=</span> <span class="n">clGetContextInfo</span><span class="p">(</span><span class="n">context</span><span class="p">,</span> <span class="n">CL_CONTEXT_DEVICES</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="nb">NULL</span><span class="p">,</span> <span class="o">&amp;</span><span class="n">deviceBufferSize</span><span class="p">);</span></span>
<span id="LC92" class="line">    <span class="k">if</span> <span class="p">(</span><span class="n">errNum</span> <span class="o">!=</span> <span class="n">CL_SUCCESS</span><span class="p">)</span></span>
<span id="LC93" class="line">    <span class="p">{</span></span>
<span id="LC94" class="line">        <span class="n">std</span><span class="o">::</span><span class="n">cerr</span> <span class="o">&lt;&lt;</span> <span class="s">"Failed call to clGetContextInfo(...,GL_CONTEXT_DEVICES,...)"</span><span class="p">;</span></span>
<span id="LC95" class="line">        <span class="k">return</span> <span class="nb">NULL</span><span class="p">;</span></span>
<span id="LC96" class="line">    <span class="p">}</span></span>
<span id="LC97" class="line"></span>
<span id="LC98" class="line">    <span class="k">if</span> <span class="p">(</span><span class="n">deviceBufferSize</span> <span class="o">&lt;=</span> <span class="mi">0</span><span class="p">)</span></span>
<span id="LC99" class="line">    <span class="p">{</span></span>
<span id="LC100" class="line">        <span class="n">std</span><span class="o">::</span><span class="n">cerr</span> <span class="o">&lt;&lt;</span> <span class="s">"No devices available."</span><span class="p">;</span></span>
<span id="LC101" class="line">        <span class="k">return</span> <span class="nb">NULL</span><span class="p">;</span></span>
<span id="LC102" class="line">    <span class="p">}</span></span>
<span id="LC103" class="line"></span>
<span id="LC104" class="line">    <span class="c1">// Allocate memory for the devices buffer</span></span>
<span id="LC105" class="line">    <span class="n">devices</span> <span class="o">=</span> <span class="k">new</span> <span class="n">cl_device_id</span><span class="p">[</span><span class="n">deviceBufferSize</span> <span class="o">/</span> <span class="k">sizeof</span><span class="p">(</span><span class="n">cl_device_id</span><span class="p">)];</span></span>
<span id="LC106" class="line">    <span class="n">errNum</span> <span class="o">=</span> <span class="n">clGetContextInfo</span><span class="p">(</span><span class="n">context</span><span class="p">,</span> <span class="n">CL_CONTEXT_DEVICES</span><span class="p">,</span> <span class="n">deviceBufferSize</span><span class="p">,</span> <span class="n">devices</span><span class="p">,</span> <span class="nb">NULL</span><span class="p">);</span></span>
<span id="LC107" class="line">    <span class="k">if</span> <span class="p">(</span><span class="n">errNum</span> <span class="o">!=</span> <span class="n">CL_SUCCESS</span><span class="p">)</span></span>
<span id="LC108" class="line">    <span class="p">{</span></span>
<span id="LC109" class="line">        <span class="n">std</span><span class="o">::</span><span class="n">cerr</span> <span class="o">&lt;&lt;</span> <span class="s">"Failed to get device IDs"</span><span class="p">;</span></span>
<span id="LC110" class="line">        <span class="k">return</span> <span class="nb">NULL</span><span class="p">;</span></span>
<span id="LC111" class="line">    <span class="p">}</span></span>
<span id="LC112" class="line"></span>
<span id="LC113" class="line">    <span class="c1">// In this example, we just choose the first available device.  In a</span></span>
<span id="LC114" class="line">    <span class="c1">// real program, you would likely use all available devices or choose</span></span>
<span id="LC115" class="line">    <span class="c1">// the highest performance device based on OpenCL device queries</span></span>
<span id="LC116" class="line">    <span class="n">commandQueue</span> <span class="o">=</span> <span class="n">clCreateCommandQueue</span><span class="p">(</span><span class="n">context</span><span class="p">,</span> <span class="n">devices</span><span class="p">[</span><span class="mi">0</span><span class="p">],</span> <span class="mi">0</span><span class="p">,</span> <span class="nb">NULL</span><span class="p">);</span></span>
<span id="LC117" class="line">    <span class="k">if</span> <span class="p">(</span><span class="n">commandQueue</span> <span class="o">==</span> <span class="nb">NULL</span><span class="p">)</span></span>
<span id="LC118" class="line">    <span class="p">{</span></span>
<span id="LC119" class="line">        <span class="n">std</span><span class="o">::</span><span class="n">cerr</span> <span class="o">&lt;&lt;</span> <span class="s">"Failed to create commandQueue for device 0"</span><span class="p">;</span></span>
<span id="LC120" class="line">        <span class="k">return</span> <span class="nb">NULL</span><span class="p">;</span></span>
<span id="LC121" class="line">    <span class="p">}</span></span>
<span id="LC122" class="line"></span>
<span id="LC123" class="line">    <span class="o">*</span><span class="n">device</span> <span class="o">=</span> <span class="n">devices</span><span class="p">[</span><span class="mi">0</span><span class="p">];</span></span>
<span id="LC124" class="line">    <span class="k">delete</span> <span class="p">[]</span> <span class="n">devices</span><span class="p">;</span></span>
<span id="LC125" class="line">    <span class="k">return</span> <span class="n">commandQueue</span><span class="p">;</span></span>
<span id="LC126" class="line"><span class="p">}</span></span>
<span id="LC127" class="line"></span>
<span id="LC128" class="line"><span class="c1">///</span></span>
<span id="LC129" class="line"><span class="c1">//  Create an OpenCL program from the kernel source file</span></span>
<span id="LC130" class="line"><span class="c1">//</span></span>
<span id="LC131" class="line"><span class="n">cl_program</span> <span class="nf">CreateProgram</span><span class="p">(</span><span class="n">cl_context</span> <span class="n">context</span><span class="p">,</span> <span class="n">cl_device_id</span> <span class="n">device</span><span class="p">,</span> <span class="k">const</span> <span class="kt">char</span><span class="o">*</span> <span class="n">fileName</span><span class="p">)</span></span>
<span id="LC132" class="line"><span class="p">{</span></span>
<span id="LC133" class="line">    <span class="n">cl_int</span> <span class="n">errNum</span><span class="p">;</span></span>
<span id="LC134" class="line">    <span class="n">cl_program</span> <span class="n">program</span><span class="p">;</span></span>
<span id="LC135" class="line"></span>
<span id="LC136" class="line">    <span class="n">std</span><span class="o">::</span><span class="n">ifstream</span> <span class="n">kernelFile</span><span class="p">(</span><span class="n">fileName</span><span class="p">,</span> <span class="n">std</span><span class="o">::</span><span class="n">ios</span><span class="o">::</span><span class="n">in</span><span class="p">);</span></span>
<span id="LC137" class="line">    <span class="k">if</span> <span class="p">(</span><span class="o">!</span><span class="n">kernelFile</span><span class="p">.</span><span class="n">is_open</span><span class="p">())</span></span>
<span id="LC138" class="line">    <span class="p">{</span></span>
<span id="LC139" class="line">        <span class="n">std</span><span class="o">::</span><span class="n">cerr</span> <span class="o">&lt;&lt;</span> <span class="s">"Failed to open file for reading: "</span> <span class="o">&lt;&lt;</span> <span class="n">fileName</span> <span class="o">&lt;&lt;</span> <span class="n">std</span><span class="o">::</span><span class="n">endl</span><span class="p">;</span></span>
<span id="LC140" class="line">        <span class="k">return</span> <span class="nb">NULL</span><span class="p">;</span></span>
<span id="LC141" class="line">    <span class="p">}</span></span>
<span id="LC142" class="line"></span>
<span id="LC143" class="line">    <span class="n">std</span><span class="o">::</span><span class="n">ostringstream</span> <span class="n">oss</span><span class="p">;</span></span>
<span id="LC144" class="line">    <span class="n">oss</span> <span class="o">&lt;&lt;</span> <span class="n">kernelFile</span><span class="p">.</span><span class="n">rdbuf</span><span class="p">();</span></span>
<span id="LC145" class="line"></span>
<span id="LC146" class="line">    <span class="n">std</span><span class="o">::</span><span class="n">string</span> <span class="n">srcStdStr</span> <span class="o">=</span> <span class="n">oss</span><span class="p">.</span><span class="n">str</span><span class="p">();</span></span>
<span id="LC147" class="line">    <span class="k">const</span> <span class="kt">char</span> <span class="o">*</span><span class="n">srcStr</span> <span class="o">=</span> <span class="n">srcStdStr</span><span class="p">.</span><span class="n">c_str</span><span class="p">();</span></span>
<span id="LC148" class="line">    <span class="n">program</span> <span class="o">=</span> <span class="n">clCreateProgramWithSource</span><span class="p">(</span><span class="n">context</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span></span>
<span id="LC149" class="line">                                        <span class="p">(</span><span class="k">const</span> <span class="kt">char</span><span class="o">**</span><span class="p">)</span><span class="o">&amp;</span><span class="n">srcStr</span><span class="p">,</span></span>
<span id="LC150" class="line">                                        <span class="nb">NULL</span><span class="p">,</span> <span class="nb">NULL</span><span class="p">);</span></span>
<span id="LC151" class="line">    <span class="k">if</span> <span class="p">(</span><span class="n">program</span> <span class="o">==</span> <span class="nb">NULL</span><span class="p">)</span></span>
<span id="LC152" class="line">    <span class="p">{</span></span>
<span id="LC153" class="line">        <span class="n">std</span><span class="o">::</span><span class="n">cerr</span> <span class="o">&lt;&lt;</span> <span class="s">"Failed to create CL program from source."</span> <span class="o">&lt;&lt;</span> <span class="n">std</span><span class="o">::</span><span class="n">endl</span><span class="p">;</span></span>
<span id="LC154" class="line">        <span class="k">return</span> <span class="nb">NULL</span><span class="p">;</span></span>
<span id="LC155" class="line">    <span class="p">}</span></span>
<span id="LC156" class="line"></span>
<span id="LC157" class="line">    <span class="n">errNum</span> <span class="o">=</span> <span class="n">clBuildProgram</span><span class="p">(</span><span class="n">program</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="nb">NULL</span><span class="p">,</span> <span class="nb">NULL</span><span class="p">,</span> <span class="nb">NULL</span><span class="p">,</span> <span class="nb">NULL</span><span class="p">);</span></span>
<span id="LC158" class="line">    <span class="k">if</span> <span class="p">(</span><span class="n">errNum</span> <span class="o">!=</span> <span class="n">CL_SUCCESS</span><span class="p">)</span></span>
<span id="LC159" class="line">    <span class="p">{</span></span>
<span id="LC160" class="line">        <span class="c1">// Determine the reason for the error</span></span>
<span id="LC161" class="line">        <span class="kt">char</span> <span class="n">buildLog</span><span class="p">[</span><span class="mi">16384</span><span class="p">];</span></span>
<span id="LC162" class="line">        <span class="n">clGetProgramBuildInfo</span><span class="p">(</span><span class="n">program</span><span class="p">,</span> <span class="n">device</span><span class="p">,</span> <span class="n">CL_PROGRAM_BUILD_LOG</span><span class="p">,</span></span>
<span id="LC163" class="line">                              <span class="k">sizeof</span><span class="p">(</span><span class="n">buildLog</span><span class="p">),</span> <span class="n">buildLog</span><span class="p">,</span> <span class="nb">NULL</span><span class="p">);</span></span>
<span id="LC164" class="line"></span>
<span id="LC165" class="line">        <span class="n">std</span><span class="o">::</span><span class="n">cerr</span> <span class="o">&lt;&lt;</span> <span class="s">"Error in kernel: "</span> <span class="o">&lt;&lt;</span> <span class="n">std</span><span class="o">::</span><span class="n">endl</span><span class="p">;</span></span>
<span id="LC166" class="line">        <span class="n">std</span><span class="o">::</span><span class="n">cerr</span> <span class="o">&lt;&lt;</span> <span class="n">buildLog</span><span class="p">;</span></span>
<span id="LC167" class="line">        <span class="n">clReleaseProgram</span><span class="p">(</span><span class="n">program</span><span class="p">);</span></span>
<span id="LC168" class="line">        <span class="k">return</span> <span class="nb">NULL</span><span class="p">;</span></span>
<span id="LC169" class="line">    <span class="p">}</span></span>
<span id="LC170" class="line"></span>
<span id="LC171" class="line">    <span class="k">return</span> <span class="n">program</span><span class="p">;</span></span>
<span id="LC172" class="line"><span class="p">}</span></span>
<span id="LC173" class="line"></span>
<span id="LC174" class="line"></span>
<span id="LC175" class="line"><span class="c1">///</span></span>
<span id="LC176" class="line"><span class="c1">//  Cleanup any created OpenCL resources</span></span>
<span id="LC177" class="line"><span class="c1">//</span></span>
<span id="LC178" class="line"><span class="kt">void</span> <span class="nf">Cleanup</span><span class="p">(</span><span class="n">cl_context</span> <span class="n">context</span><span class="p">,</span> <span class="n">cl_command_queue</span> <span class="n">commandQueue</span><span class="p">,</span></span>
<span id="LC179" class="line">             <span class="n">cl_program</span> <span class="n">program</span><span class="p">,</span> <span class="n">cl_kernel</span> <span class="n">kernel</span><span class="p">,</span> <span class="n">cl_mem</span> <span class="n">imageObjects</span><span class="p">[</span><span class="mi">2</span><span class="p">],</span></span>
<span id="LC180" class="line">             <span class="n">cl_sampler</span> <span class="n">sampler</span><span class="p">)</span></span>
<span id="LC181" class="line"><span class="p">{</span></span>
<span id="LC182" class="line">    <span class="k">for</span> <span class="p">(</span><span class="kt">int</span> <span class="n">i</span> <span class="o">=</span> <span class="mi">0</span><span class="p">;</span> <span class="n">i</span> <span class="o">&lt;</span> <span class="mi">2</span><span class="p">;</span> <span class="n">i</span><span class="o">++</span><span class="p">)</span></span>
<span id="LC183" class="line">    <span class="p">{</span></span>
<span id="LC184" class="line">        <span class="k">if</span> <span class="p">(</span><span class="n">imageObjects</span><span class="p">[</span><span class="n">i</span><span class="p">]</span> <span class="o">!=</span> <span class="mi">0</span><span class="p">)</span></span>
<span id="LC185" class="line">            <span class="n">clReleaseMemObject</span><span class="p">(</span><span class="n">imageObjects</span><span class="p">[</span><span class="n">i</span><span class="p">]);</span></span>
<span id="LC186" class="line">    <span class="p">}</span></span>
<span id="LC187" class="line">    <span class="k">if</span> <span class="p">(</span><span class="n">commandQueue</span> <span class="o">!=</span> <span class="mi">0</span><span class="p">)</span></span>
<span id="LC188" class="line">        <span class="n">clReleaseCommandQueue</span><span class="p">(</span><span class="n">commandQueue</span><span class="p">);</span></span>
<span id="LC189" class="line"></span>
<span id="LC190" class="line">    <span class="k">if</span> <span class="p">(</span><span class="n">kernel</span> <span class="o">!=</span> <span class="mi">0</span><span class="p">)</span></span>
<span id="LC191" class="line">        <span class="n">clReleaseKernel</span><span class="p">(</span><span class="n">kernel</span><span class="p">);</span></span>
<span id="LC192" class="line"></span>
<span id="LC193" class="line">    <span class="k">if</span> <span class="p">(</span><span class="n">program</span> <span class="o">!=</span> <span class="mi">0</span><span class="p">)</span></span>
<span id="LC194" class="line">        <span class="n">clReleaseProgram</span><span class="p">(</span><span class="n">program</span><span class="p">);</span></span>
<span id="LC195" class="line"></span>
<span id="LC196" class="line">    <span class="k">if</span> <span class="p">(</span><span class="n">sampler</span> <span class="o">!=</span> <span class="mi">0</span><span class="p">)</span></span>
<span id="LC197" class="line">        <span class="n">clReleaseSampler</span><span class="p">(</span><span class="n">sampler</span><span class="p">);</span></span>
<span id="LC198" class="line"></span>
<span id="LC199" class="line">    <span class="k">if</span> <span class="p">(</span><span class="n">context</span> <span class="o">!=</span> <span class="mi">0</span><span class="p">)</span></span>
<span id="LC200" class="line">        <span class="n">clReleaseContext</span><span class="p">(</span><span class="n">context</span><span class="p">);</span></span>
<span id="LC201" class="line"></span>
<span id="LC202" class="line"><span class="p">}</span></span>
<span id="LC203" class="line"></span>
<span id="LC204" class="line"><span class="c1">///</span></span>
<span id="LC205" class="line"><span class="c1">//  Load an image using the FreeImage library and create an OpenCL</span></span>
<span id="LC206" class="line"><span class="c1">//  image out of it</span></span>
<span id="LC207" class="line"><span class="c1">//</span></span>
<span id="LC208" class="line"><span class="n">cl_mem</span> <span class="nf">LoadImage</span><span class="p">(</span><span class="n">cl_context</span> <span class="n">context</span><span class="p">,</span> <span class="kt">char</span> <span class="o">*</span><span class="n">fileName</span><span class="p">,</span> <span class="kt">int</span> <span class="o">&amp;</span><span class="n">width</span><span class="p">,</span> <span class="kt">int</span> <span class="o">&amp;</span><span class="n">height</span><span class="p">)</span></span>
<span id="LC209" class="line"><span class="p">{</span></span>
<span id="LC210" class="line">    <span class="n">FREE_IMAGE_FORMAT</span> <span class="n">format</span> <span class="o">=</span> <span class="n">FreeImage_GetFileType</span><span class="p">(</span><span class="n">fileName</span><span class="p">,</span> <span class="mi">0</span><span class="p">);</span></span>
<span id="LC211" class="line">    <span class="n">FIBITMAP</span><span class="o">*</span> <span class="n">image</span> <span class="o">=</span> <span class="n">FreeImage_Load</span><span class="p">(</span><span class="n">format</span><span class="p">,</span> <span class="n">fileName</span><span class="p">);</span></span>
<span id="LC212" class="line"></span>
<span id="LC213" class="line">    <span class="c1">// Convert to 32-bit image</span></span>
<span id="LC214" class="line">    <span class="n">FIBITMAP</span><span class="o">*</span> <span class="n">temp</span> <span class="o">=</span> <span class="n">image</span><span class="p">;</span></span>
<span id="LC215" class="line">    <span class="n">image</span> <span class="o">=</span> <span class="n">FreeImage_ConvertTo32Bits</span><span class="p">(</span><span class="n">image</span><span class="p">);</span></span>
<span id="LC216" class="line">    <span class="n">FreeImage_Unload</span><span class="p">(</span><span class="n">temp</span><span class="p">);</span></span>
<span id="LC217" class="line"></span>
<span id="LC218" class="line">    <span class="n">width</span> <span class="o">=</span> <span class="n">FreeImage_GetWidth</span><span class="p">(</span><span class="n">image</span><span class="p">);</span></span>
<span id="LC219" class="line">    <span class="n">height</span> <span class="o">=</span> <span class="n">FreeImage_GetHeight</span><span class="p">(</span><span class="n">image</span><span class="p">);</span></span>
<span id="LC220" class="line"></span>
<span id="LC221" class="line">    <span class="kt">char</span> <span class="o">*</span><span class="n">buffer</span> <span class="o">=</span> <span class="k">new</span> <span class="kt">char</span><span class="p">[</span><span class="n">width</span> <span class="o">*</span> <span class="n">height</span> <span class="o">*</span> <span class="mi">4</span><span class="p">];</span></span>
<span id="LC222" class="line">    <span class="n">memcpy</span><span class="p">(</span><span class="n">buffer</span><span class="p">,</span> <span class="n">FreeImage_GetBits</span><span class="p">(</span><span class="n">image</span><span class="p">),</span> <span class="n">width</span> <span class="o">*</span> <span class="n">height</span> <span class="o">*</span> <span class="mi">4</span><span class="p">);</span></span>
<span id="LC223" class="line"></span>
<span id="LC224" class="line">    <span class="n">FreeImage_Unload</span><span class="p">(</span><span class="n">image</span><span class="p">);</span></span>
<span id="LC225" class="line"></span>
<span id="LC226" class="line">    <span class="c1">// Create OpenCL image</span></span>
<span id="LC227" class="line">    <span class="n">cl_image_format</span> <span class="n">clImageFormat</span><span class="p">;</span></span>
<span id="LC228" class="line">    <span class="n">clImageFormat</span><span class="p">.</span><span class="n">image_channel_order</span> <span class="o">=</span> <span class="n">CL_RGBA</span><span class="p">;</span></span>
<span id="LC229" class="line">    <span class="n">clImageFormat</span><span class="p">.</span><span class="n">image_channel_data_type</span> <span class="o">=</span> <span class="n">CL_UNORM_INT8</span><span class="p">;</span></span>
<span id="LC230" class="line"></span>
<span id="LC231" class="line">    <span class="n">cl_int</span> <span class="n">errNum</span><span class="p">;</span></span>
<span id="LC232" class="line">    <span class="n">cl_mem</span> <span class="n">clImage</span><span class="p">;</span></span>
<span id="LC233" class="line">    <span class="n">clImage</span> <span class="o">=</span> <span class="n">clCreateImage2D</span><span class="p">(</span><span class="n">context</span><span class="p">,</span></span>
<span id="LC234" class="line">                            <span class="n">CL_MEM_READ_ONLY</span> <span class="o">|</span> <span class="n">CL_MEM_COPY_HOST_PTR</span><span class="p">,</span></span>
<span id="LC235" class="line">                            <span class="o">&amp;</span><span class="n">clImageFormat</span><span class="p">,</span></span>
<span id="LC236" class="line">                            <span class="n">width</span><span class="p">,</span></span>
<span id="LC237" class="line">                            <span class="n">height</span><span class="p">,</span></span>
<span id="LC238" class="line">                            <span class="mi">0</span><span class="p">,</span></span>
<span id="LC239" class="line">                            <span class="n">buffer</span><span class="p">,</span></span>
<span id="LC240" class="line">                            <span class="o">&amp;</span><span class="n">errNum</span><span class="p">);</span></span>
<span id="LC241" class="line"></span>
<span id="LC242" class="line">    <span class="k">if</span> <span class="p">(</span><span class="n">errNum</span> <span class="o">!=</span> <span class="n">CL_SUCCESS</span><span class="p">)</span></span>
<span id="LC243" class="line">    <span class="p">{</span></span>
<span id="LC244" class="line">        <span class="n">std</span><span class="o">::</span><span class="n">cerr</span> <span class="o">&lt;&lt;</span> <span class="s">"Error creating CL image object"</span> <span class="o">&lt;&lt;</span> <span class="n">std</span><span class="o">::</span><span class="n">endl</span><span class="p">;</span></span>
<span id="LC245" class="line">        <span class="k">return</span> <span class="mi">0</span><span class="p">;</span></span>
<span id="LC246" class="line">    <span class="p">}</span></span>
<span id="LC247" class="line"></span>
<span id="LC248" class="line">    <span class="k">return</span> <span class="n">clImage</span><span class="p">;</span></span>
<span id="LC249" class="line"><span class="p">}</span></span>
<span id="LC250" class="line"></span>
<span id="LC251" class="line"><span class="c1">///</span></span>
<span id="LC252" class="line"><span class="c1">//  Save an image using the FreeImage library</span></span>
<span id="LC253" class="line"><span class="c1">//</span></span>
<span id="LC254" class="line"><span class="kt">bool</span> <span class="nf">SaveImage</span><span class="p">(</span><span class="kt">char</span> <span class="o">*</span><span class="n">fileName</span><span class="p">,</span> <span class="kt">char</span> <span class="o">*</span><span class="n">buffer</span><span class="p">,</span> <span class="kt">int</span> <span class="n">width</span><span class="p">,</span> <span class="kt">int</span> <span class="n">height</span><span class="p">)</span></span>
<span id="LC255" class="line"><span class="p">{</span></span>
<span id="LC256" class="line">    <span class="n">FREE_IMAGE_FORMAT</span> <span class="n">format</span> <span class="o">=</span> <span class="n">FreeImage_GetFIFFromFilename</span><span class="p">(</span><span class="n">fileName</span><span class="p">);</span></span>
<span id="LC257" class="line">    <span class="n">FIBITMAP</span> <span class="o">*</span><span class="n">image</span> <span class="o">=</span> <span class="n">FreeImage_ConvertFromRawBits</span><span class="p">((</span><span class="n">BYTE</span><span class="o">*</span><span class="p">)</span><span class="n">buffer</span><span class="p">,</span> <span class="n">width</span><span class="p">,</span></span>
<span id="LC258" class="line">                        <span class="n">height</span><span class="p">,</span> <span class="n">width</span> <span class="o">*</span> <span class="mi">4</span><span class="p">,</span> <span class="mi">32</span><span class="p">,</span></span>
<span id="LC259" class="line">                        <span class="mh">0xFF000000</span><span class="p">,</span> <span class="mh">0x00FF0000</span><span class="p">,</span> <span class="mh">0x0000FF00</span><span class="p">);</span></span>
<span id="LC260" class="line">    <span class="k">return</span> <span class="p">(</span><span class="n">FreeImage_Save</span><span class="p">(</span><span class="n">format</span><span class="p">,</span> <span class="n">image</span><span class="p">,</span> <span class="n">fileName</span><span class="p">)</span> <span class="o">==</span> <span class="n">TRUE</span><span class="p">)</span> <span class="o">?</span> <span class="nb">true</span> <span class="o">:</span> <span class="nb">false</span><span class="p">;</span></span>
<span id="LC261" class="line"><span class="p">}</span></span>
<span id="LC262" class="line"></span>
<span id="LC263" class="line"><span class="c1">///</span></span>
<span id="LC264" class="line"><span class="c1">//  Round up to the nearest multiple of the group size</span></span>
<span id="LC265" class="line"><span class="c1">//</span></span>
<span id="LC266" class="line"><span class="kt">size_t</span> <span class="nf">RoundUp</span><span class="p">(</span><span class="kt">int</span> <span class="n">groupSize</span><span class="p">,</span> <span class="kt">int</span> <span class="n">globalSize</span><span class="p">)</span></span>
<span id="LC267" class="line"><span class="p">{</span></span>
<span id="LC268" class="line">    <span class="kt">int</span> <span class="n">r</span> <span class="o">=</span> <span class="n">globalSize</span> <span class="o">%</span> <span class="n">groupSize</span><span class="p">;</span></span>
<span id="LC269" class="line">    <span class="k">if</span><span class="p">(</span><span class="n">r</span> <span class="o">==</span> <span class="mi">0</span><span class="p">)</span></span>
<span id="LC270" class="line">    <span class="p">{</span></span>
<span id="LC271" class="line">     	<span class="k">return</span> <span class="n">globalSize</span><span class="p">;</span></span>
<span id="LC272" class="line">    <span class="p">}</span></span>
<span id="LC273" class="line">    <span class="k">else</span></span>
<span id="LC274" class="line">    <span class="p">{</span></span>
<span id="LC275" class="line">     	<span class="k">return</span> <span class="n">globalSize</span> <span class="o">+</span> <span class="n">groupSize</span> <span class="o">-</span> <span class="n">r</span><span class="p">;</span></span>
<span id="LC276" class="line">    <span class="p">}</span></span>
<span id="LC277" class="line"><span class="p">}</span></span>
<span id="LC278" class="line"></span>
<span id="LC279" class="line"><span class="c1">///</span></span>
<span id="LC280" class="line"><span class="c1">//	main() for HelloBinaryWorld example</span></span>
<span id="LC281" class="line"><span class="c1">//</span></span>
<span id="LC282" class="line"><span class="kt">int</span> <span class="nf">main</span><span class="p">(</span><span class="kt">int</span> <span class="n">argc</span><span class="p">,</span> <span class="kt">char</span><span class="o">**</span> <span class="n">argv</span><span class="p">)</span></span>
<span id="LC283" class="line"><span class="p">{</span></span>
<span id="LC284" class="line">    <span class="n">cl_context</span> <span class="n">context</span> <span class="o">=</span> <span class="mi">0</span><span class="p">;</span></span>
<span id="LC285" class="line">    <span class="n">cl_command_queue</span> <span class="n">commandQueue</span> <span class="o">=</span> <span class="mi">0</span><span class="p">;</span></span>
<span id="LC286" class="line">    <span class="n">cl_program</span> <span class="n">program</span> <span class="o">=</span> <span class="mi">0</span><span class="p">;</span></span>
<span id="LC287" class="line">    <span class="n">cl_device_id</span> <span class="n">device</span> <span class="o">=</span> <span class="mi">0</span><span class="p">;</span></span>
<span id="LC288" class="line">    <span class="n">cl_kernel</span> <span class="n">kernel</span> <span class="o">=</span> <span class="mi">0</span><span class="p">;</span></span>
<span id="LC289" class="line">    <span class="n">cl_mem</span> <span class="n">imageObjects</span><span class="p">[</span><span class="mi">2</span><span class="p">]</span> <span class="o">=</span> <span class="p">{</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span> <span class="p">};</span></span>
<span id="LC290" class="line">    <span class="n">cl_sampler</span> <span class="n">sampler</span> <span class="o">=</span> <span class="mi">0</span><span class="p">;</span></span>
<span id="LC291" class="line">    <span class="n">cl_int</span> <span class="n">errNum</span><span class="p">;</span></span>
<span id="LC292" class="line"></span>
<span id="LC293" class="line"></span>
<span id="LC294" class="line">    <span class="k">if</span> <span class="p">(</span><span class="n">argc</span> <span class="o">!=</span> <span class="mi">3</span><span class="p">)</span></span>
<span id="LC295" class="line">    <span class="p">{</span></span>
<span id="LC296" class="line">        <span class="n">std</span><span class="o">::</span><span class="n">cerr</span> <span class="o">&lt;&lt;</span> <span class="s">"USAGE: "</span> <span class="o">&lt;&lt;</span> <span class="n">argv</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span> <span class="o">&lt;&lt;</span> <span class="s">" &lt;inputImageFile&gt; &lt;outputImageFiles&gt;"</span> <span class="o">&lt;&lt;</span> <span class="n">std</span><span class="o">::</span><span class="n">endl</span><span class="p">;</span></span>
<span id="LC297" class="line">        <span class="k">return</span> <span class="mi">1</span><span class="p">;</span></span>
<span id="LC298" class="line">    <span class="p">}</span></span>
<span id="LC299" class="line"></span>
<span id="LC300" class="line">    <span class="c1">// Create an OpenCL context on first available platform</span></span>
<span id="LC301" class="line">    <span class="n">context</span> <span class="o">=</span> <span class="n">CreateContext</span><span class="p">();</span></span>
<span id="LC302" class="line">    <span class="k">if</span> <span class="p">(</span><span class="n">context</span> <span class="o">==</span> <span class="nb">NULL</span><span class="p">)</span></span>
<span id="LC303" class="line">    <span class="p">{</span></span>
<span id="LC304" class="line">        <span class="n">std</span><span class="o">::</span><span class="n">cerr</span> <span class="o">&lt;&lt;</span> <span class="s">"Failed to create OpenCL context."</span> <span class="o">&lt;&lt;</span> <span class="n">std</span><span class="o">::</span><span class="n">endl</span><span class="p">;</span></span>
<span id="LC305" class="line">        <span class="k">return</span> <span class="mi">1</span><span class="p">;</span></span>
<span id="LC306" class="line">    <span class="p">}</span></span>
<span id="LC307" class="line"></span>
<span id="LC308" class="line">    <span class="c1">// Create a command-queue on the first device available</span></span>
<span id="LC309" class="line">    <span class="c1">// on the created context</span></span>
<span id="LC310" class="line">    <span class="n">commandQueue</span> <span class="o">=</span> <span class="n">CreateCommandQueue</span><span class="p">(</span><span class="n">context</span><span class="p">,</span> <span class="o">&amp;</span><span class="n">device</span><span class="p">);</span></span>
<span id="LC311" class="line">    <span class="k">if</span> <span class="p">(</span><span class="n">commandQueue</span> <span class="o">==</span> <span class="nb">NULL</span><span class="p">)</span></span>
<span id="LC312" class="line">    <span class="p">{</span></span>
<span id="LC313" class="line">        <span class="n">Cleanup</span><span class="p">(</span><span class="n">context</span><span class="p">,</span> <span class="n">commandQueue</span><span class="p">,</span> <span class="n">program</span><span class="p">,</span> <span class="n">kernel</span><span class="p">,</span> <span class="n">imageObjects</span><span class="p">,</span> <span class="n">sampler</span><span class="p">);</span></span>
<span id="LC314" class="line">        <span class="k">return</span> <span class="mi">1</span><span class="p">;</span></span>
<span id="LC315" class="line">    <span class="p">}</span></span>
<span id="LC316" class="line"></span>
<span id="LC317" class="line">    <span class="c1">// Make sure the device supports images, otherwise exit</span></span>
<span id="LC318" class="line">    <span class="n">cl_bool</span> <span class="n">imageSupport</span> <span class="o">=</span> <span class="n">CL_FALSE</span><span class="p">;</span></span>
<span id="LC319" class="line">    <span class="n">clGetDeviceInfo</span><span class="p">(</span><span class="n">device</span><span class="p">,</span> <span class="n">CL_DEVICE_IMAGE_SUPPORT</span><span class="p">,</span> <span class="k">sizeof</span><span class="p">(</span><span class="n">cl_bool</span><span class="p">),</span></span>
<span id="LC320" class="line">                    <span class="o">&amp;</span><span class="n">imageSupport</span><span class="p">,</span> <span class="nb">NULL</span><span class="p">);</span></span>
<span id="LC321" class="line">    <span class="k">if</span> <span class="p">(</span><span class="n">imageSupport</span> <span class="o">!=</span> <span class="n">CL_TRUE</span><span class="p">)</span></span>
<span id="LC322" class="line">    <span class="p">{</span></span>
<span id="LC323" class="line">        <span class="n">std</span><span class="o">::</span><span class="n">cerr</span> <span class="o">&lt;&lt;</span> <span class="s">"OpenCL device does not support images."</span> <span class="o">&lt;&lt;</span> <span class="n">std</span><span class="o">::</span><span class="n">endl</span><span class="p">;</span></span>
<span id="LC324" class="line">        <span class="n">Cleanup</span><span class="p">(</span><span class="n">context</span><span class="p">,</span> <span class="n">commandQueue</span><span class="p">,</span> <span class="n">program</span><span class="p">,</span> <span class="n">kernel</span><span class="p">,</span> <span class="n">imageObjects</span><span class="p">,</span> <span class="n">sampler</span><span class="p">);</span></span>
<span id="LC325" class="line">        <span class="k">return</span> <span class="mi">1</span><span class="p">;</span></span>
<span id="LC326" class="line">    <span class="p">}</span></span>
<span id="LC327" class="line"></span>
<span id="LC328" class="line">    <span class="c1">// Load input image from file and load it into</span></span>
<span id="LC329" class="line">    <span class="c1">// an OpenCL image object</span></span>
<span id="LC330" class="line">    <span class="kt">int</span> <span class="n">width</span><span class="p">,</span> <span class="n">height</span><span class="p">;</span></span>
<span id="LC331" class="line">    <span class="n">imageObjects</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span> <span class="o">=</span> <span class="n">LoadImage</span><span class="p">(</span><span class="n">context</span><span class="p">,</span> <span class="n">argv</span><span class="p">[</span><span class="mi">1</span><span class="p">],</span> <span class="n">width</span><span class="p">,</span> <span class="n">height</span><span class="p">);</span></span>
<span id="LC332" class="line">    <span class="k">if</span> <span class="p">(</span><span class="n">imageObjects</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span> <span class="o">==</span> <span class="mi">0</span><span class="p">)</span></span>
<span id="LC333" class="line">    <span class="p">{</span></span>
<span id="LC334" class="line">        <span class="n">std</span><span class="o">::</span><span class="n">cerr</span> <span class="o">&lt;&lt;</span> <span class="s">"Error loading: "</span> <span class="o">&lt;&lt;</span> <span class="n">std</span><span class="o">::</span><span class="n">string</span><span class="p">(</span><span class="n">argv</span><span class="p">[</span><span class="mi">1</span><span class="p">])</span> <span class="o">&lt;&lt;</span> <span class="n">std</span><span class="o">::</span><span class="n">endl</span><span class="p">;</span></span>
<span id="LC335" class="line">        <span class="n">Cleanup</span><span class="p">(</span><span class="n">context</span><span class="p">,</span> <span class="n">commandQueue</span><span class="p">,</span> <span class="n">program</span><span class="p">,</span> <span class="n">kernel</span><span class="p">,</span> <span class="n">imageObjects</span><span class="p">,</span> <span class="n">sampler</span><span class="p">);</span></span>
<span id="LC336" class="line">        <span class="k">return</span> <span class="mi">1</span><span class="p">;</span></span>
<span id="LC337" class="line">    <span class="p">}</span></span>
<span id="LC338" class="line"></span>
<span id="LC339" class="line">    <span class="c1">// Create ouput image object</span></span>
<span id="LC340" class="line">    <span class="n">cl_image_format</span> <span class="n">clImageFormat</span><span class="p">;</span></span>
<span id="LC341" class="line">    <span class="n">clImageFormat</span><span class="p">.</span><span class="n">image_channel_order</span> <span class="o">=</span> <span class="n">CL_RGBA</span><span class="p">;</span></span>
<span id="LC342" class="line">    <span class="n">clImageFormat</span><span class="p">.</span><span class="n">image_channel_data_type</span> <span class="o">=</span> <span class="n">CL_UNORM_INT8</span><span class="p">;</span></span>
<span id="LC343" class="line">    <span class="n">imageObjects</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span> <span class="o">=</span> <span class="n">clCreateImage2D</span><span class="p">(</span><span class="n">context</span><span class="p">,</span></span>
<span id="LC344" class="line">                                       <span class="n">CL_MEM_WRITE_ONLY</span><span class="p">,</span></span>
<span id="LC345" class="line">                                       <span class="o">&amp;</span><span class="n">clImageFormat</span><span class="p">,</span></span>
<span id="LC346" class="line">                                       <span class="n">width</span><span class="p">,</span></span>
<span id="LC347" class="line">                                       <span class="n">height</span><span class="p">,</span></span>
<span id="LC348" class="line">                                       <span class="mi">0</span><span class="p">,</span></span>
<span id="LC349" class="line">                                       <span class="nb">NULL</span><span class="p">,</span></span>
<span id="LC350" class="line">                                       <span class="o">&amp;</span><span class="n">errNum</span><span class="p">);</span></span>
<span id="LC351" class="line"></span>
<span id="LC352" class="line">    <span class="k">if</span> <span class="p">(</span><span class="n">errNum</span> <span class="o">!=</span> <span class="n">CL_SUCCESS</span><span class="p">)</span></span>
<span id="LC353" class="line">    <span class="p">{</span></span>
<span id="LC354" class="line">        <span class="n">std</span><span class="o">::</span><span class="n">cerr</span> <span class="o">&lt;&lt;</span> <span class="s">"Error creating CL output image object."</span> <span class="o">&lt;&lt;</span> <span class="n">std</span><span class="o">::</span><span class="n">endl</span><span class="p">;</span></span>
<span id="LC355" class="line">        <span class="n">Cleanup</span><span class="p">(</span><span class="n">context</span><span class="p">,</span> <span class="n">commandQueue</span><span class="p">,</span> <span class="n">program</span><span class="p">,</span> <span class="n">kernel</span><span class="p">,</span> <span class="n">imageObjects</span><span class="p">,</span> <span class="n">sampler</span><span class="p">);</span></span>
<span id="LC356" class="line">        <span class="k">return</span> <span class="mi">1</span><span class="p">;</span></span>
<span id="LC357" class="line">    <span class="p">}</span></span>
<span id="LC358" class="line"></span>
<span id="LC359" class="line"></span>
<span id="LC360" class="line">    <span class="c1">// Create sampler for sampling image object</span></span>
<span id="LC361" class="line">    <span class="n">sampler</span> <span class="o">=</span> <span class="n">clCreateSampler</span><span class="p">(</span><span class="n">context</span><span class="p">,</span></span>
<span id="LC362" class="line">                              <span class="n">CL_FALSE</span><span class="p">,</span> <span class="c1">// Non-normalized coordinates</span></span>
<span id="LC363" class="line">                              <span class="n">CL_ADDRESS_CLAMP_TO_EDGE</span><span class="p">,</span></span>
<span id="LC364" class="line">                              <span class="n">CL_FILTER_NEAREST</span><span class="p">,</span></span>
<span id="LC365" class="line">                              <span class="o">&amp;</span><span class="n">errNum</span><span class="p">);</span></span>
<span id="LC366" class="line"></span>
<span id="LC367" class="line">    <span class="k">if</span> <span class="p">(</span><span class="n">errNum</span> <span class="o">!=</span> <span class="n">CL_SUCCESS</span><span class="p">)</span></span>
<span id="LC368" class="line">    <span class="p">{</span></span>
<span id="LC369" class="line">        <span class="n">std</span><span class="o">::</span><span class="n">cerr</span> <span class="o">&lt;&lt;</span> <span class="s">"Error creating CL sampler object."</span> <span class="o">&lt;&lt;</span> <span class="n">std</span><span class="o">::</span><span class="n">endl</span><span class="p">;</span></span>
<span id="LC370" class="line">        <span class="n">Cleanup</span><span class="p">(</span><span class="n">context</span><span class="p">,</span> <span class="n">commandQueue</span><span class="p">,</span> <span class="n">program</span><span class="p">,</span> <span class="n">kernel</span><span class="p">,</span> <span class="n">imageObjects</span><span class="p">,</span> <span class="n">sampler</span><span class="p">);</span></span>
<span id="LC371" class="line">        <span class="k">return</span> <span class="mi">1</span><span class="p">;</span></span>
<span id="LC372" class="line">    <span class="p">}</span></span>
<span id="LC373" class="line"></span>
<span id="LC374" class="line">    <span class="c1">// Create OpenCL program</span></span>
<span id="LC375" class="line">    <span class="n">program</span> <span class="o">=</span> <span class="n">CreateProgram</span><span class="p">(</span><span class="n">context</span><span class="p">,</span> <span class="n">device</span><span class="p">,</span> <span class="s">"ImageFilter2D.cl"</span><span class="p">);</span></span>
<span id="LC376" class="line">    <span class="k">if</span> <span class="p">(</span><span class="n">program</span> <span class="o">==</span> <span class="nb">NULL</span><span class="p">)</span></span>
<span id="LC377" class="line">    <span class="p">{</span></span>
<span id="LC378" class="line">        <span class="n">Cleanup</span><span class="p">(</span><span class="n">context</span><span class="p">,</span> <span class="n">commandQueue</span><span class="p">,</span> <span class="n">program</span><span class="p">,</span> <span class="n">kernel</span><span class="p">,</span> <span class="n">imageObjects</span><span class="p">,</span> <span class="n">sampler</span><span class="p">);</span></span>
<span id="LC379" class="line">        <span class="k">return</span> <span class="mi">1</span><span class="p">;</span></span>
<span id="LC380" class="line">    <span class="p">}</span></span>
<span id="LC381" class="line"></span>
<span id="LC382" class="line">    <span class="c1">// Create OpenCL kernel</span></span>
<span id="LC383" class="line">    <span class="n">kernel</span> <span class="o">=</span> <span class="n">clCreateKernel</span><span class="p">(</span><span class="n">program</span><span class="p">,</span> <span class="s">"gaussian_filter"</span><span class="p">,</span> <span class="nb">NULL</span><span class="p">);</span></span>
<span id="LC384" class="line">    <span class="k">if</span> <span class="p">(</span><span class="n">kernel</span> <span class="o">==</span> <span class="nb">NULL</span><span class="p">)</span></span>
<span id="LC385" class="line">    <span class="p">{</span></span>
<span id="LC386" class="line">        <span class="n">std</span><span class="o">::</span><span class="n">cerr</span> <span class="o">&lt;&lt;</span> <span class="s">"Failed to create kernel"</span> <span class="o">&lt;&lt;</span> <span class="n">std</span><span class="o">::</span><span class="n">endl</span><span class="p">;</span></span>
<span id="LC387" class="line">        <span class="n">Cleanup</span><span class="p">(</span><span class="n">context</span><span class="p">,</span> <span class="n">commandQueue</span><span class="p">,</span> <span class="n">program</span><span class="p">,</span> <span class="n">kernel</span><span class="p">,</span> <span class="n">imageObjects</span><span class="p">,</span> <span class="n">sampler</span><span class="p">);</span></span>
<span id="LC388" class="line">        <span class="k">return</span> <span class="mi">1</span><span class="p">;</span></span>
<span id="LC389" class="line">    <span class="p">}</span></span>
<span id="LC390" class="line"></span>
<span id="LC391" class="line">    <span class="c1">// Set the kernel arguments</span></span>
<span id="LC392" class="line">    <span class="n">errNum</span> <span class="o">=</span> <span class="n">clSetKernelArg</span><span class="p">(</span><span class="n">kernel</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="k">sizeof</span><span class="p">(</span><span class="n">cl_mem</span><span class="p">),</span> <span class="o">&amp;</span><span class="n">imageObjects</span><span class="p">[</span><span class="mi">0</span><span class="p">]);</span></span>
<span id="LC393" class="line">    <span class="n">errNum</span> <span class="o">|=</span> <span class="n">clSetKernelArg</span><span class="p">(</span><span class="n">kernel</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="k">sizeof</span><span class="p">(</span><span class="n">cl_mem</span><span class="p">),</span> <span class="o">&amp;</span><span class="n">imageObjects</span><span class="p">[</span><span class="mi">1</span><span class="p">]);</span></span>
<span id="LC394" class="line">    <span class="n">errNum</span> <span class="o">|=</span> <span class="n">clSetKernelArg</span><span class="p">(</span><span class="n">kernel</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="k">sizeof</span><span class="p">(</span><span class="n">cl_sampler</span><span class="p">),</span> <span class="o">&amp;</span><span class="n">sampler</span><span class="p">);</span></span>
<span id="LC395" class="line">    <span class="n">errNum</span> <span class="o">|=</span> <span class="n">clSetKernelArg</span><span class="p">(</span><span class="n">kernel</span><span class="p">,</span> <span class="mi">3</span><span class="p">,</span> <span class="k">sizeof</span><span class="p">(</span><span class="n">cl_int</span><span class="p">),</span> <span class="o">&amp;</span><span class="n">width</span><span class="p">);</span></span>
<span id="LC396" class="line">    <span class="n">errNum</span> <span class="o">|=</span> <span class="n">clSetKernelArg</span><span class="p">(</span><span class="n">kernel</span><span class="p">,</span> <span class="mi">4</span><span class="p">,</span> <span class="k">sizeof</span><span class="p">(</span><span class="n">cl_int</span><span class="p">),</span> <span class="o">&amp;</span><span class="n">height</span><span class="p">);</span></span>
<span id="LC397" class="line">    <span class="k">if</span> <span class="p">(</span><span class="n">errNum</span> <span class="o">!=</span> <span class="n">CL_SUCCESS</span><span class="p">)</span></span>
<span id="LC398" class="line">    <span class="p">{</span></span>
<span id="LC399" class="line">        <span class="n">std</span><span class="o">::</span><span class="n">cerr</span> <span class="o">&lt;&lt;</span> <span class="s">"Error setting kernel arguments."</span> <span class="o">&lt;&lt;</span> <span class="n">std</span><span class="o">::</span><span class="n">endl</span><span class="p">;</span></span>
<span id="LC400" class="line">        <span class="n">Cleanup</span><span class="p">(</span><span class="n">context</span><span class="p">,</span> <span class="n">commandQueue</span><span class="p">,</span> <span class="n">program</span><span class="p">,</span> <span class="n">kernel</span><span class="p">,</span> <span class="n">imageObjects</span><span class="p">,</span> <span class="n">sampler</span><span class="p">);</span></span>
<span id="LC401" class="line">        <span class="k">return</span> <span class="mi">1</span><span class="p">;</span></span>
<span id="LC402" class="line">    <span class="p">}</span></span>
<span id="LC403" class="line"></span>
<span id="LC404" class="line">    <span class="kt">size_t</span> <span class="n">localWorkSize</span><span class="p">[</span><span class="mi">2</span><span class="p">]</span> <span class="o">=</span> <span class="p">{</span> <span class="mi">16</span><span class="p">,</span> <span class="mi">16</span> <span class="p">};</span></span>
<span id="LC405" class="line">    <span class="kt">size_t</span> <span class="n">globalWorkSize</span><span class="p">[</span><span class="mi">2</span><span class="p">]</span> <span class="o">=</span>  <span class="p">{</span> <span class="n">RoundUp</span><span class="p">(</span><span class="n">localWorkSize</span><span class="p">[</span><span class="mi">0</span><span class="p">],</span> <span class="n">width</span><span class="p">),</span></span>
<span id="LC406" class="line">                                  <span class="n">RoundUp</span><span class="p">(</span><span class="n">localWorkSize</span><span class="p">[</span><span class="mi">1</span><span class="p">],</span> <span class="n">height</span><span class="p">)</span> <span class="p">};</span></span>
<span id="LC407" class="line"></span>
<span id="LC408" class="line">    <span class="c1">// Queue the kernel up for execution</span></span>
<span id="LC409" class="line">    <span class="n">errNum</span> <span class="o">=</span> <span class="n">clEnqueueNDRangeKernel</span><span class="p">(</span><span class="n">commandQueue</span><span class="p">,</span> <span class="n">kernel</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="nb">NULL</span><span class="p">,</span></span>
<span id="LC410" class="line">                                    <span class="n">globalWorkSize</span><span class="p">,</span> <span class="n">localWorkSize</span><span class="p">,</span></span>
<span id="LC411" class="line">                                    <span class="mi">0</span><span class="p">,</span> <span class="nb">NULL</span><span class="p">,</span> <span class="nb">NULL</span><span class="p">);</span></span>
<span id="LC412" class="line">    <span class="k">if</span> <span class="p">(</span><span class="n">errNum</span> <span class="o">!=</span> <span class="n">CL_SUCCESS</span><span class="p">)</span></span>
<span id="LC413" class="line">    <span class="p">{</span></span>
<span id="LC414" class="line">        <span class="n">std</span><span class="o">::</span><span class="n">cerr</span> <span class="o">&lt;&lt;</span> <span class="s">"Error queuing kernel for execution."</span> <span class="o">&lt;&lt;</span> <span class="n">std</span><span class="o">::</span><span class="n">endl</span><span class="p">;</span></span>
<span id="LC415" class="line">        <span class="n">Cleanup</span><span class="p">(</span><span class="n">context</span><span class="p">,</span> <span class="n">commandQueue</span><span class="p">,</span> <span class="n">program</span><span class="p">,</span> <span class="n">kernel</span><span class="p">,</span> <span class="n">imageObjects</span><span class="p">,</span> <span class="n">sampler</span><span class="p">);</span></span>
<span id="LC416" class="line">        <span class="k">return</span> <span class="mi">1</span><span class="p">;</span></span>
<span id="LC417" class="line">    <span class="p">}</span></span>
<span id="LC418" class="line"></span>
<span id="LC419" class="line">    <span class="c1">// Read the output buffer back to the Host</span></span>
<span id="LC420" class="line">    <span class="kt">char</span> <span class="o">*</span><span class="n">buffer</span> <span class="o">=</span> <span class="k">new</span> <span class="kt">char</span> <span class="p">[</span><span class="n">width</span> <span class="o">*</span> <span class="n">height</span> <span class="o">*</span> <span class="mi">4</span><span class="p">];</span></span>
<span id="LC421" class="line">    <span class="kt">size_t</span> <span class="n">origin</span><span class="p">[</span><span class="mi">3</span><span class="p">]</span> <span class="o">=</span> <span class="p">{</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span> <span class="p">};</span></span>
<span id="LC422" class="line">    <span class="kt">size_t</span> <span class="n">region</span><span class="p">[</span><span class="mi">3</span><span class="p">]</span> <span class="o">=</span> <span class="p">{</span> <span class="n">width</span><span class="p">,</span> <span class="n">height</span><span class="p">,</span> <span class="mi">1</span><span class="p">};</span></span>
<span id="LC423" class="line">    <span class="n">errNum</span> <span class="o">=</span> <span class="n">clEnqueueReadImage</span><span class="p">(</span><span class="n">commandQueue</span><span class="p">,</span> <span class="n">imageObjects</span><span class="p">[</span><span class="mi">1</span><span class="p">],</span> <span class="n">CL_TRUE</span><span class="p">,</span></span>
<span id="LC424" class="line">                                <span class="n">origin</span><span class="p">,</span> <span class="n">region</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="n">buffer</span><span class="p">,</span></span>
<span id="LC425" class="line">                                <span class="mi">0</span><span class="p">,</span> <span class="nb">NULL</span><span class="p">,</span> <span class="nb">NULL</span><span class="p">);</span></span>
<span id="LC426" class="line">    <span class="k">if</span> <span class="p">(</span><span class="n">errNum</span> <span class="o">!=</span> <span class="n">CL_SUCCESS</span><span class="p">)</span></span>
<span id="LC427" class="line">    <span class="p">{</span></span>
<span id="LC428" class="line">        <span class="n">std</span><span class="o">::</span><span class="n">cerr</span> <span class="o">&lt;&lt;</span> <span class="s">"Error reading result buffer."</span> <span class="o">&lt;&lt;</span> <span class="n">std</span><span class="o">::</span><span class="n">endl</span><span class="p">;</span></span>
<span id="LC429" class="line">        <span class="n">Cleanup</span><span class="p">(</span><span class="n">context</span><span class="p">,</span> <span class="n">commandQueue</span><span class="p">,</span> <span class="n">program</span><span class="p">,</span> <span class="n">kernel</span><span class="p">,</span> <span class="n">imageObjects</span><span class="p">,</span> <span class="n">sampler</span><span class="p">);</span></span>
<span id="LC430" class="line">        <span class="k">return</span> <span class="mi">1</span><span class="p">;</span></span>
<span id="LC431" class="line">    <span class="p">}</span></span>
<span id="LC432" class="line"></span>
<span id="LC433" class="line">    <span class="n">std</span><span class="o">::</span><span class="n">cout</span> <span class="o">&lt;&lt;</span> <span class="n">std</span><span class="o">::</span><span class="n">endl</span><span class="p">;</span></span>
<span id="LC434" class="line">    <span class="n">std</span><span class="o">::</span><span class="n">cout</span> <span class="o">&lt;&lt;</span> <span class="s">"Executed program succesfully."</span> <span class="o">&lt;&lt;</span> <span class="n">std</span><span class="o">::</span><span class="n">endl</span><span class="p">;</span></span>
<span id="LC435" class="line"></span>
<span id="LC436" class="line">    <span class="c1">//memset(buffer, 0xff, width * height * 4);</span></span>
<span id="LC437" class="line">    <span class="c1">// Save the image out to disk</span></span>
<span id="LC438" class="line">    <span class="k">if</span> <span class="p">(</span><span class="o">!</span><span class="n">SaveImage</span><span class="p">(</span><span class="n">argv</span><span class="p">[</span><span class="mi">2</span><span class="p">],</span> <span class="n">buffer</span><span class="p">,</span> <span class="n">width</span><span class="p">,</span> <span class="n">height</span><span class="p">))</span></span>
<span id="LC439" class="line">    <span class="p">{</span></span>
<span id="LC440" class="line">        <span class="n">std</span><span class="o">::</span><span class="n">cerr</span> <span class="o">&lt;&lt;</span> <span class="s">"Error writing output image: "</span> <span class="o">&lt;&lt;</span> <span class="n">argv</span><span class="p">[</span><span class="mi">2</span><span class="p">]</span> <span class="o">&lt;&lt;</span> <span class="n">std</span><span class="o">::</span><span class="n">endl</span><span class="p">;</span></span>
<span id="LC441" class="line">        <span class="n">Cleanup</span><span class="p">(</span><span class="n">context</span><span class="p">,</span> <span class="n">commandQueue</span><span class="p">,</span> <span class="n">program</span><span class="p">,</span> <span class="n">kernel</span><span class="p">,</span> <span class="n">imageObjects</span><span class="p">,</span> <span class="n">sampler</span><span class="p">);</span></span>
<span id="LC442" class="line">        <span class="k">delete</span> <span class="p">[]</span> <span class="n">buffer</span><span class="p">;</span></span>
<span id="LC443" class="line">        <span class="k">return</span> <span class="mi">1</span><span class="p">;</span></span>
<span id="LC444" class="line">    <span class="p">}</span></span>
<span id="LC445" class="line"></span>
<span id="LC446" class="line">    <span class="k">delete</span> <span class="p">[]</span> <span class="n">buffer</span><span class="p">;</span></span>
<span id="LC447" class="line">    <span class="n">Cleanup</span><span class="p">(</span><span class="n">context</span><span class="p">,</span> <span class="n">commandQueue</span><span class="p">,</span> <span class="n">program</span><span class="p">,</span> <span class="n">kernel</span><span class="p">,</span> <span class="n">imageObjects</span><span class="p">,</span> <span class="n">sampler</span><span class="p">);</span></span>
<span id="LC448" class="line">    <span class="k">return</span> <span class="mi">0</span><span class="p">;</span></span>
<span id="LC449" class="line"><span class="p">}</span></span></code></pre>
</div>
</div>


</article>
</div>

</div>
<div class="modal" id="modal-remove-blob">
<div class="modal-dialog">
<div class="modal-content">
<div class="modal-header">
<a class="close" data-dismiss="modal" href="#">×</a>
<h3 class="page-title">Delete ImageFilter2D.cpp</h3>
</div>
<div class="modal-body">
<form class="form-horizontal js-replace-blob-form js-quick-submit js-requires-input" action="/EN-605.417.31_FA16/intro_to_gpu/blob/master/opencl-book-samples/src/Chapter_8/ImageFilter2D/ImageFilter2D.cpp" accept-charset="UTF-8" method="post"><input name="utf8" type="hidden" value="&#x2713;" /><input type="hidden" name="_method" value="delete" /><input type="hidden" name="authenticity_token" value="CjnDwWoz8EaIFt4YgsahB7bfVsfgtLL8SnIQpWVKC9xGCGBLITZh0SwTMvdFlsw0rQaBFRp/kd61GbEkXINiyw==" /><div class="form-group commit_message-group">
<label class="control-label" for="commit_message-6582579ea672ab15eecefd5cc34cf51e">Commit message
</label><div class="col-sm-10">
<div class="commit-message-container">
<div class="max-width-marker"></div>
<textarea name="commit_message" id="commit_message-6582579ea672ab15eecefd5cc34cf51e" class="form-control js-commit-message" placeholder="Delete ImageFilter2D.cpp" required="required" rows="3">
Delete ImageFilter2D.cpp</textarea>
</div>
</div>
</div>

<div class="form-group branch">
<label class="control-label" for="target_branch">Target branch</label>
<div class="col-sm-10">
<input type="text" name="target_branch" id="target_branch" value="master" required="required" class="form-control js-target-branch" />
<div class="js-create-merge-request-container">
<div class="checkbox">
<label for="create_merge_request-d23e5c0734fdb1ef9fe48fcb251913a4"><input type="checkbox" name="create_merge_request" id="create_merge_request-d23e5c0734fdb1ef9fe48fcb251913a4" value="1" class="js-create-merge-request" checked="checked" />
Start a <strong>new merge request</strong> with these changes
</label></div>
</div>
</div>
</div>
<input type="hidden" name="original_branch" id="original_branch" value="master" class="js-original-branch" />

<div class="form-group">
<div class="col-sm-offset-2 col-sm-10">
<button name="button" type="submit" class="btn btn-remove btn-remove-file">Delete file</button>
<a class="btn btn-cancel" data-dismiss="modal" href="#">Cancel</a>
</div>
</div>
</form></div>
</div>
</div>
</div>
<script>
  new NewCommitForm($('.js-replace-blob-form'))
</script>

<div class="modal" id="modal-upload-blob">
<div class="modal-dialog">
<div class="modal-content">
<div class="modal-header">
<a class="close" data-dismiss="modal" href="#">×</a>
<h3 class="page-title">Replace ImageFilter2D.cpp</h3>
</div>
<div class="modal-body">
<form class="js-quick-submit js-upload-blob-form form-horizontal" action="/EN-605.417.31_FA16/intro_to_gpu/update/master/opencl-book-samples/src/Chapter_8/ImageFilter2D/ImageFilter2D.cpp" accept-charset="UTF-8" method="post"><input name="utf8" type="hidden" value="&#x2713;" /><input type="hidden" name="_method" value="put" /><input type="hidden" name="authenticity_token" value="ipuR2wdr3e7kXA9PgzpgCHfut91y68HjijybUp6JFbLGqjJRTG5MeUBZ46BEag07bDdgD4gg4sF1VzrTp0B8pQ==" /><div class="dropzone">
<div class="dropzone-previews blob-upload-dropzone-previews">
<p class="dz-message light">
Attach a file by drag &amp; drop or
<a class="markdown-selector" href="#">click to upload</a>
</p>
</div>
</div>
<br>
<div class="alert alert-danger data dropzone-alerts" style="display:none"></div>
<div class="form-group commit_message-group">
<label class="control-label" for="commit_message-c73333923163fe84a0d33e51a7e98f8e">Commit message
</label><div class="col-sm-10">
<div class="commit-message-container">
<div class="max-width-marker"></div>
<textarea name="commit_message" id="commit_message-c73333923163fe84a0d33e51a7e98f8e" class="form-control js-commit-message" placeholder="Replace ImageFilter2D.cpp" required="required" rows="3">
Replace ImageFilter2D.cpp</textarea>
</div>
</div>
</div>

<div class="form-group branch">
<label class="control-label" for="target_branch">Target branch</label>
<div class="col-sm-10">
<input type="text" name="target_branch" id="target_branch" value="master" required="required" class="form-control js-target-branch" />
<div class="js-create-merge-request-container">
<div class="checkbox">
<label for="create_merge_request-f51e98666400d50bc086eeb41fc500e1"><input type="checkbox" name="create_merge_request" id="create_merge_request-f51e98666400d50bc086eeb41fc500e1" value="1" class="js-create-merge-request" checked="checked" />
Start a <strong>new merge request</strong> with these changes
</label></div>
</div>
</div>
</div>
<input type="hidden" name="original_branch" id="original_branch" value="master" class="js-original-branch" />

<div class="form-actions">
<button name="button" type="submit" class="btn btn-small btn-create btn-upload-file" id="submit-all">Replace file</button>
<a class="btn btn-cancel" data-dismiss="modal" href="#">Cancel</a>
</div>
</form></div>
</div>
</div>
</div>
<script>
  gl.utils.disableButtonIfEmptyField($('.js-upload-blob-form').find('.js-commit-message'), '.btn-upload-file');
  new BlobFileDropzone($('.js-upload-blob-form'), 'put');
  new NewCommitForm($('.js-upload-blob-form'))
</script>

</div>

</div>
</div>
</div>
</div>



</body>
</html>

