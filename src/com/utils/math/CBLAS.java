package com.utils.math;

public final class CBLAS {
	static {
		System.loadLibrary("com_utils_math_CBLAS");
	}
	
	public final static class ORDER {
		private ORDER() {}
		
		public final static int RowMajor = 101;
		public final static int ColMajor = 102;
	}
	
	public final static class TRANSPOSE {
		private TRANSPOSE() {}
		
		public final static int NoTrans   = 111;
		public final static int Trans     = 112;
		public final static int ConjTrans = 113;
	}
	
	public static native void dgemm(int Order, int TransA, int TransB, int M, int N, int K, double alpha, double[] A, int lda, double[] B, int ldb, double beta, double[] C, int ldc);
}
