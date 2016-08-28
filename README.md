# AutoEncoder
DeepAutoEncoder&amp;DeepAutoClassifier

Java implimention of the paper 《Reducing The Dimensionality Of Data With Neural Networks》. <br />

# Execution
1. com.autoencoder.DeepAutoEncoder      <---> Encode Image
2. com.autoencoder.DeepAutoClassifier   <---> Image Classification

# Results
add soon.

# Update
2016/8/28: Uing Intel® MKL to accelerate matrix multiplication. <br />
Referenced web page: https://software.intel.com/en-us/articles/using-intel-mkl-in-math-intensive-java-applications-on-intel-xeon-phi <br />


# Cautious
1. Project Error <br />
  Maybe the file "./lib/eugfc.jar" can not be found or JDK version is not right.

2. Exception in thread "main" java.lang.UnsatisfiedLinkError: no com_utils_math_CBLAS in java.library.path <br />
  Build Path -> Configure Build Path -> Java Build Path -> Libraries -> JRE System Library -> Native library location <br />
  Then, select the exec folder(the folder contain "com_utils_math_CBLAS.dll").
