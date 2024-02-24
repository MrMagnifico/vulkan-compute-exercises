# Vulkan Compute Exercises
## Implementation Description
- `00_MyGPU`: Vulkan application initialisation and basic information readout for utilised GPU
- `01_HelloGPU`: Usage of debug `printf` to print the ID of each working GPU thread
- `02_UniformBuffer`: Usage of a uniform buffer for basic data transmission
- `03_StorageBuffer`: Usage of a storage buffer to compute the Fibonnaci sequence
- `04_Copying`: Utilisation of buffer transfer operations
- `05_EdgeDetector`: Sobel edge detector for 2D images (operates on a per color channel basis)
- `06_Atomic`: Usage of atomic operations to compute prime numbers up to a given limit
- `07_PointClouderRender`: Point cloud renderer utilising the 'Basic Approach' from [Schütz et al.](https://www.cg.tuwien.ac.at/research/publications/2021/SCHUETZ-2021-PCC/)
- `08_SharedMemory`: Usage of shared memory to compute the Fibonnaci sequence
- `09_MatrixMultiplication`: Parallel matrix-matrix multiplication [using shared memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)
- `10_Reduction`: Parallel addition reduction using [the techniques presented by Mark Harris](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf) (implements the optimisations listed until 'Kernel 4')
- `11_StagingBuffer`: Same as `10_Reduction` but using a staging buffer to transfer data to and from the GPU in order to bypass constraints on device local, host visible memory

## Original README
This framework provides the code base for completing a suite of basic Vulkan compute exercises. To build it, you will need
- A GPU that is capable of running compute shaders with up-to-date drivers (support for Vulkan 1.3)
- The Vulkan SDK (fetch it from LunarG, https://www.lunarg.com/vulkan-sdk/)
- Install Vulkan with VMA
- A recent version of CMake (https://cmake.org/)
- An up-to-date C++ compiler (e.g., Microsoft Visual Studio C++ Compiler)
- Optional: if you want to debug shaders, get RenderDoc (fetch it at https://renderdoc.org/)

### CMake Instructions
- Install the above requirements and clone the repository. Run CMAKE (ideally via the GUI).
- Pick the compiler that you intend to use.
- If you plan to debug shaders with RenderDoc, enable `DEBUG_SHADERS`. You will need to point CMake to the RenderDoc directory.
- As source directory, pick the repo directory.
- Choose a target directory, e.g., `<repo_directory>/build`. It is possible to use the source directory as the target, but this is bad practice.
- Hit `Configure`.
- If everything worked (no errors), hit `Generate`.
- You should now have your compiler's preferred solution for building projects (`.sln` on windows or Makefile on Linux).

### Build Instructions
Select the project you want to work on as the startup target. In Visual Studio, this is done by right-clicking a project and 'Set as Startup Project'. After this, it will be compiled and run when you hit the "Play" button. The framework already includes a custom build step that automatically compiles your shader files along with your C++ code. It will put the compiled SPIR-V files in the project directory. On Windows, this means when you run one of the assignment applications from Visual Studio, it should just find the compiled shader files. If it does not, you might have to adjust your IDE's working directory (check slides). The debugging of shaders is independently controlled by the `DEBUG_SHADERS` option in your CMake settings. Shader debugging code paths should only trigger if RenderDoc is active simultaneously. 

### Task Instructions
Each task comes with a `main.cpp`, which provides the instructions that you should follow. Vulkan is quite verbose, and there will be basic steps that we will be doing a lot (creating instances, devices, etc.). Hence, there is a dedicated file `framework.h` and `framework.cpp` where these reusable code parts should go. The tasks will at some times instruct you to write solutions in `main.cpp`, and other times to implement basic functionality in `framework.cpp`. There are also compute shader files, ending with `.comp`, which you will need to modify with GLSL code. They should show up in your project's list of source files. 

**Don't forget to activate your validation layers via the Vulkan Configurator before you start developing!** If you need `printf` debugging (required for Task 1), make sure that this is enabled as well.

### Disclaimer
This is an early version, provided to collect opinions and subject to later extension. Suggestions and corrections are welcome!
