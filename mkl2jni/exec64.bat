cl /I "C:/Program Files/Java/jdk1.8.0_77/include" /I "C:/Program Files/Java/jdk1.8.0_77/include/win32" /I "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries_2016.3.207/windows/mkl/include" /I "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries_2016.3.207/windows/mkl/include/intel64" /I "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries_2016.3.207/windows/mkl/include/fftw" -LD com_utils_math_CBLAS.c mkl_core.lib mkl_intel_thread.lib libiomp5md.lib mkl_intel_ilp64.lib -Fecom_utils_math_CBLAS.dll /link /libpath:"C:/Program Files (x86)/IntelSWTools/compilers_and_libraries_2016.3.207/windows/mkl/lib/intel64_win" /libpath:"C:/Program Files (x86)/IntelSWTools/compilers_and_libraries_2016.3.207/windows/compiler/lib/intel64_win"
