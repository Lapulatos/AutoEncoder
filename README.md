# AutoEncoder
DeepAutoEncoder&amp;DeepAutoClassifier

Java implimention of the paper 《Reducing The Dimensionality Of Data With Neural Networks》. <br />

# Execution
1. com.autoencoder.DeepAutoEncoder      <---> Encode Image
2. com.autoencoder.DeepAutoClassifier   <---> Image Classification

# Results
1. Encode Image: <br />
<img border="0" src=https://github.com/Lapulatos/AutoEncoder/blob/master/encoder.png> <br />
  This is the encode result of the Deep Auto Encoder(encode 10 times in total). In each time, the result will be greater than the before.

# Update
2016/8/28: Uing Intel® MKL to accelerate matrix multiplication. <br />
Referenced web page: https://software.intel.com/en-us/articles/using-intel-mkl-in-math-intensive-java-applications-on-intel-xeon-phi <br />


# Cautious
1. Project Error <br />
  Maybe the file "./lib/eugfc.jar" can not be found or JDK version is not right.

2. Exception in thread "main" java.lang.UnsatisfiedLinkError: no com_utils_math_CBLAS in java.library.path <br />
  Build Path -> Configure Build Path -> Java Build Path -> Libraries -> JRE System Library -> Native library location <br />
  Then, select the exec folder(the folder contain "com_utils_math_CBLAS.dll").

3. Linux Platform Compile Error <br />
  I compiled the c source code on Ubuntu 16.04 LTS, and my compile parameters are: <br />
    gcc -shared -fPIC -I. -I$JAVA_HOME/include -I$JAVA_HOME/include/linux -I$MKL_HOME/mkl/include -L$MKL_HOME/mkl/lib/intel64 -L$MKL_HOME/compiler/lib/intel64 com_utils_math_CBLAS.c -o libmkl_java_utils.so -lm -lpthread -ldl -lmkl_core -lmkl_intel_thread -lmkl_rt <br />
  The last three parameters can not exchange the order.(It seems like this, I don't know why. =_=)
