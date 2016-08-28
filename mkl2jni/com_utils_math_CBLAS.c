#include <jni.h>
#include <assert.h>

#include "mkl.h"

#include "com_utils_math_CBLAS.h"

JNIEXPORT void JNICALL Java_com_utils_math_CBLAS_dgemm(JNIEnv * env, 
							jclass  kclass, 
							jint	Order, 
							jint	TransA, 
							jint	TransB, 
							jint	M, 
							jint	N, 
							jint	K, 
							jdouble	alpha, 
							jdoubleArray	A, 
							jint		lda, 
							jdoubleArray	B, 
							jint		ldb, 
							jdouble		beta, 
							jdoubleArray	C, 
							jint		ldc) {
	jdouble *aElems, *bElems, *cElems;

	aElems = (*env)->GetDoubleArrayElements(env, A, NULL);
	bElems = (*env)->GetDoubleArrayElements(env, B, NULL);
	cElems = (*env)->GetDoubleArrayElements(env, C, NULL);

	assert(aElems && bElems && cElems);

	cblas_dgemm((CBLAS_ORDER)Order, (CBLAS_TRANSPOSE)TransA, (CBLAS_TRANSPOSE)TransB, (int)M, (int)N, (int)K, (double)alpha, aElems, (int)lda, bElems, (int)ldb, beta, cElems, (int)ldc);	

	(*env)->ReleaseDoubleArrayElements(env, C, cElems, 0);
	(*env)->ReleaseDoubleArrayElements(env, B, bElems, JNI_ABORT);
	(*env)->ReleaseDoubleArrayElements(env, A, aElems, JNI_ABORT);
}
