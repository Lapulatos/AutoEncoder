package com.utils.math;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.Random;

import com.utils.ByteArrayToDataArray;
import com.utils.ByteToInteger;
import com.utils.DataArrayToByteArray;

public class Matrix {

	public int m, n;
	public double[] data;
	
	public Matrix() {
		m = n = 0;
		data = null;
	}
	
	public Matrix(int mm, int nn) {
		if(mm < 1 || nn < 1) {
			System.out.println("params 'm' or 'n' is under 1, which is invalid for the constructor Matrix(int, int).");
			return;
		}
		this.m = mm; this.n = nn;
		data = new double[mm*nn];
	}
	
	public Matrix(int mm, int nn, double v) {
		if(mm < 1 || nn < 1) {
			System.out.println("params 'm' or 'n' is under 1, which is invalid for the constructor Matrix(int, int, int).");
			return;
		}
		this.m = mm; this.n = nn;
		data = new double[mm*nn];
		
		for(int i = 0; i < mm*nn; ++i) {
			data[i] = v;
		}
	}
	
	public Matrix(Matrix mtx) {
		m = mtx.m;
		n = mtx.n;
		
		data = new double[m*n];
		for(int i = 0; i < m*n; ++i) {
			data[i] = mtx.data[i];
		}
	}
	
	public Matrix(double[] res, int mm, int nn) {
		if(mm < 1 || nn < 1) {
			System.out.println("params 'mm' or 'nn' is under 1, which is invalid for the constructor Matrix(double[], int, int).");
			return;
		}
		data = res;
		m = mm;
		n = nn;
	}
	
	public Matrix(int[] res, int mm, int nn) {
		if(mm < 1 || nn < 1) {
			System.out.println("params 'mm' or 'nn' is under 1, which is invalid for the constructor Matrix(int[], int, int).");
			return;
		}

		m = mm;
		n = nn;
		data = new double[mm*nn];
		
		for(int i = 0; i < mm*nn; ++i) {
			data[i] = (double)res[i];
		}
	}
	
	public Matrix(int mm, int nn, int initVal, int step) {
		if(m < 1 || nn < 1 || step < 1) {
			System.out.println("params 'mm' or 'nn' or step is under 1, which is invalid for the constructor Matrix(int, int, int, int).");
			return;
		}
		
		m = mm;
		n = nn;
		data = new double[mm*nn];
		
		data[0] = initVal;
		for(int i = 1; i < mm*nn; ++i) {
			data[i] = data[i-1]+step;
		}
	}
	
	public static Matrix[] normalizeMatrix(Matrix[] mtx, double scale) {
		if(Math.abs(scale) < Math.pow(10, -7)) {
			System.out.println("scale is nearly equal to zero. normalizeMatrix(Matrix[], double)");
			return null;
		}
		
		Matrix[] res = new Matrix[mtx.length];
		
		for(int k = 0; k < mtx.length; ++k) {
			for(int i = 0; i < mtx[0].data.length; ++i) {
				res[k].data[i] = mtx[k].data[i] / scale;
			}
		}
		
		return res;
	}
	
	public static Matrix[] createMultiMtx(int[][] I, int mm, int nn) {
		if(mm < 1 || nn < 1 || I[0].length != mm*nn) {
			System.out.println("params 'mm' or 'nn' is under 1 or size is not correct, which is invalid for the constructor Matrix(int[], int, int).");
			return null;
		}
		
		int size = I[0].length;
		for(int i = 1; i < I.length; ++i) {
			if(size != I[i].length) {
				System.out.println("each dim of matrix[] is not equal.");
				return null;
			}
		}
		
		Matrix[] res = new Matrix[I.length];
		
		for(int i = 0; i < I.length; ++i) {
			res[i] = new Matrix(mm, nn);
			
			for(int j = 0; j < I[0].length; ++j) {
				res[i].data[j] = I[i][j];
			}
		}
		
		return res;
	}
	
	public double minValue() {
		double min = data[0];
		
		for(int i = 0; i < data.length; ++i) {
			if(data[i] < min) {
				min = data[i];
			}
		}
		
		return min;
	}
	
	public static Matrix[] extendMatrixDim(Matrix mtx, int dim) {
		Matrix[] res = new Matrix[dim];
		for(int i = 0; i < dim; ++i) {
			res[i] = new Matrix(mtx);
		}
		
		return res;
	}
	
	public static Matrix[] createMultiMtx(double[][] I, int mm, int nn) {
		if(mm < 1 || nn < 1 || I[0].length != mm*nn) {
			System.out.println("params 'mm' or 'nn' is under 1 or size is not correct, which is invalid for the constructor Matrix(int[], int, int).");
			return null;
		}
		
		int size = I[0].length;
		for(int i = 1; i < I.length; ++i) {
			if(size != I[i].length) {
				System.out.println("each dim of matrix[] is not equal.");
				return null;
			}
		}
		
		Matrix[] res = new Matrix[I.length];
		
		for(int i = 0; i < I.length; ++i) {
			res[i] = new Matrix(mm, nn);
			
			for(int j = 0; j < I[0].length; ++j) {
				res[i].data[i] = I[i][j];
			}
		}
		
		return res;
	}
	
	public static double[][] getMultiMtxData(Matrix[] mtx) {
		double[][] res = new double[mtx.length][mtx[0].m*mtx[0].n];
		
		for(int i = 0; i < mtx.length; ++i) {
			res[i] = mtx[i].getDataTypeDouble();
		}
		
		return res;
	}
	
	public static Matrix ones(int m, int n) {
		if(m < 1 || n < 1) {
			System.out.println("params 'm' or 'n' is under 1, which is invalid for func ones.");
			return null;
		}
		
		Matrix mtx = new Matrix(m, n);
		
		for(int i = 0; i < m*n; ++i) {
			mtx.data[i] = 1;
		}
		
		return mtx;
	}
	
	public Matrix rowSum() {
		Matrix res = Matrix.zeros(this.m, 1);
		
		int mm = this.m;
		int nn = this.n;
		
		for(int i = 0; i < mm; ++i) {
			for(int j = 0; j < nn; ++j) {
				res.data[i] += this.data[i*n + j];
			}
		}
		
		return res;
	}
	
	public Matrix[] rowMax() {
		Matrix[] res = new Matrix[2];
		res[0] = Matrix.zeros(this.m, 1);
		res[1] = Matrix.zeros(this.m, 1);
		
		for(int i = 0; i < this.m; ++i) {
			res[0].data[i] = data[i*this.n];
			
			for(int j = 0; j < this.n; ++j) {
				if(res[0].data[i] < data[i*this.n + j]) {
					res[0].data[i] = data[i*this.n + j];
					res[1].data[i] = j;
				}
			}
		}
		
		return res;
	}
	
	public Matrix rowAdd(int md, int ms, double scale) {
		Matrix res = new Matrix(this);
		
		for(int i = 0; i < n; ++i) {
			res.data[md*n + i] += data[ms*n + i]*scale;
		}

		return res;
	}

	public Matrix rowMinus(int md, int ms, double scale) {
		Matrix res = new Matrix(this);
		
		for(int i = 0; i < n; ++i) {
			res.data[md*n + i] -= data[ms*n + i]*scale;
		}

		return res;
	}

	public Matrix columnSum() {
		Matrix res = Matrix.zeros(1, this.n);
		
		int mm = this.m;
		int nn = this.n;
		
		for(int i = 0; i < nn; ++i) {
			for(int j = 0; j < mm; ++j) {
				res.data[i] += this.data[j*n + i];
			}
		}

		return res;
	}
	
	public Matrix[] columnMax() {
		Matrix[] res = new Matrix[2];
		res[0] = Matrix.zeros(1, this.n);
		res[1] = Matrix.zeros(1, this.n);
		
		for(int i = 0; i < this.n; ++i) {
			res[0].data[i] = data[i];
			
			for(int j = 0; j < this.m; ++j) {
				if(res[0].data[i] < data[j*this.n + i]) {
					res[0].data[i] = data[j*this.n + i];
					res[1].data[i] = j;
				}
			}
		}
		
		return res;
	}
	
	public Matrix columnAdd(int nd, int ns, double scale) {
		Matrix res = new Matrix(this);
		
		for(int i = 0; i < m; ++i) {
			res.data[i*n + nd] += data[i*n + ns]*scale;
		}

		return res;
	}

	public Matrix columnMinus(int nd, int ns, double scale) {
		Matrix res = new Matrix(this);
		
		for(int i = 0; i < m; ++i) {
			res.data[i*n + nd] -= data[i*n + ns]*scale;
		}

		return res;
	}
	
	public Matrix add(double v) {
		Matrix mtx = new Matrix(m, n);
		
		for(int i = 0; i < m*n; ++i) {
			mtx.data[i] = data[i] + v;
		}
		
		return mtx;
	}
	
	public Matrix dotAdd(Matrix mtx) {
		if(mtx.m != m || mtx.n != n) {
			System.out.println("the dim of mtx is not equal to orig, which is invalid for func dotAdd.");
			return null;
		}
		
		Matrix res = new Matrix(m, n);
		
		for(int i = 0; i < m*n; ++i) {
			res.data[i] = data[i] + mtx.data[i];
		}
		
		return res;
	}
	
	public Matrix minus(double v) {
		Matrix mtx = new Matrix(m, n);
		
		for(int i = 0; i < m*n; ++i) {
			mtx.data[i] = data[i] - v;
		}
		
		return mtx;
	}
	
	public Matrix minus(Matrix mtx) {
		if(mtx.m != m || mtx.n != n) {
			System.out.println("the dim of mtx is not equal to orig, which is invalid for func minus.");
			return null;
		}
		Matrix res = new Matrix(m, n);
		
		for(int i = 0; i < m*n; ++i) {
			res.data[i] = data[i] - mtx.data[i];
		}
		
		return res;
	}
	
	public Matrix multi(double v) {
		Matrix mtx = new Matrix(m, n);
		
		for(int i = 0; i < m*n; ++i) {
			mtx.data[i] = data[i] * v;
		}
		
		return mtx;
	}
	
	public Matrix multi(Matrix mtx) {
		if(n != mtx.m) {
			System.out.println("mtx.n is not equal m, which is invalid for func multi.");
			return null;
		}
		
		Matrix res = new Matrix(m, mtx.n);
		
		for(int i = 0; i < m; ++i) {
			for(int j = 0; j < mtx.n; ++j) {
				res.data[i*mtx.n + j] = 0;
				
				for(int k = 0; k < n; ++k) {
					res.data[i*mtx.n + j] += data[i*n + k]*mtx.data[k*mtx.n + j];
				}
			}
		}
		
		return res;
	}
	
	public Matrix dotMinus(Matrix mtx) {
		if(mtx.m != m || mtx.n != n) {
			System.out.println("the dim of mtx is not equal to orig, which is invalid for func dotMinus.");
			return null;
		}
		
		Matrix res = new Matrix(m, n);
		
		for(int i = 0; i < m*n; ++i) {
			res.data[i] = data[i] - mtx.data[i];
		}
		
		return res;
	}
	
	public Matrix dotMulti(Matrix mtx) {
		if(mtx.m != m || mtx.n != n) {
			System.out.println("the dim of mtx is not equal to orig, which is invalid for func dotMulti.");
			return null;
		}
		
		Matrix res = new Matrix(m, n);
		
		for(int i = 0; i < m*n; ++i) {
			res.data[i] = data[i] * mtx.data[i];
		}
		
		return res;
	}
	
	public Matrix div(double v) {
		if(Math.abs(v) < Math.pow(10, -7)) {
			System.out.println("the param is nearly equal to zero.");
			return null;
		}
		
		Matrix mtx = new Matrix(m, n);
		
		for(int i = 0; i < m*n; ++i) {
			mtx.data[i] = data[i] / v;
		}
		
		return mtx;
	}
	
	public static Matrix[] multi(Matrix[] mtx, double v) {
		Matrix[] res = new Matrix[mtx.length];
		
		for(int i = 0; i < mtx.length; ++i) {
			res[i] = mtx[i].multi(v);
		}
		
		return res;
	}
	
	public static Matrix[] dotDiv(Matrix[] mtx, Matrix[] div) {
		if(mtx.length != div.length) {
			System.out.println("the dim of two matrix is not equal. dotDiv(Matrix[], Matrix[])");
			return null;
		}
		
		Matrix[] res = new Matrix[mtx.length];
		
		for(int i = 0; i < mtx.length; ++i) {
			res[i] = mtx[i].dotDiv(div[i]);
		}
		
		return res;
	}
	
	public Matrix dotDiv(Matrix mtx) {
		if(mtx.m != m || mtx.n != n) {
			System.out.println("the dim of mtx is not equal to orig, which is invalid for func dotDiv.");
			return null;
		}
		
		double epd = Math.pow(10, -8);
		for(int i = 0; i < m*n; ++i) {
			if(Math.abs(mtx.data[i]) < epd) {
				System.out.println("the matrix 'mtx' contain a number which is nearly equal to zero. dotDiv(Matrix)");
				return null;
			}
		}
		
		Matrix res = new Matrix(m, n);
		
		for(int i = 0; i < m*n; ++i) {
			res.data[i] = data[i] / mtx.data[i];
		}
		
		return res;
	}

	public Matrix square() {
		Matrix res = new Matrix(this);
		
		for(int i = 0; i < data.length; ++i) {
			res.data[i] = res.data[i] * res.data[i];
		}
		
		return res;
	}
	
	public double sum() {
		double res = 0.0;
		
		for(int i = 0; i < data.length; ++i) {
			res += data[i];
		}
		
		return res;
	}
	
	public Matrix exp() {
		Matrix res = new Matrix(this);
		
		for(int i = 0; i < data.length; ++i) {
			res.data[i] = Math.exp(res.data[i]);
		}
		
		return res;
	}
	
	public Matrix log() {
		Matrix res = new Matrix(this);
		
		for(int i = 0; i < data.length; ++i) {
			res.data[i] = Math.log(res.data[i]);
		}
		
		return res;
	}
	
	public static Matrix zeros(int m, int n) {
		if(m < 1 || n < 1) {
			System.out.println("the param 'm' or 'n' is under 1, which is invalid for func zeros(int, int).");
			return null;
		}
		
		Matrix mtx = new Matrix(m, n);
		
		for(int i = 0; i < m*n; ++i) {
			mtx.data[i] = 0;
		}
		
		return mtx;
	}
	
	public Matrix min() {
		Matrix res = null;

		if(m > 1) {
			res = new Matrix(1, n);
			
			double min;			
			for(int i = 0; i < n; ++i) {
				min = data[i];
				
				for(int j = 1; j < m; ++j) {
					if(data[j*n + i] < min) {
						min = data[j*n + i];
					}
				}
				res.data[i] = min;
			}
		} else if(m  == 1) {
			double min = data[0];
			for(int i = 1; i < data.length; ++i) {
				if(data[i] < min) {
					min = data[i];
				}
			}
			
			res = new Matrix(1, 1, min);
		}
		
		return res;
	}
	
	public Matrix max(double v) {
		Matrix res = new Matrix(this);
		
		for(int i = 0; i < data.length; ++i) {
			if(res.data[i] < v) {
				res.data[i] = v;
			}
		}
		
		return res;
	}
	
	public Matrix limitMax(double v) {
		Matrix res = new Matrix(this);
		
		for(int i = 0; i < data.length; ++i) {
			if(res.data[i] > v) {
				res.data[i] = v;
			}
		}
		
		return res;
	}
	
	public Matrix max() {
		Matrix res = null;

		if(m > 1) {
			res = new Matrix(1, n);
			
			double max;			
			for(int i = 0; i < n; ++i) {
				max = data[i];
				
				for(int j = 1; j < m; ++j) {
					if(data[j*n + i] > max) {
						max = data[j*n + i];
					}
				}
				res.data[i] = max;
			}
		} else if(m  == 1) {
			double max = data[0];
			for(int i = 1; i < data.length; ++i) {
				if(data[i] > max) {
					max = data[i];
				}
			}
			
			res = new Matrix(1, 1, max);
		}
		
		return res;
	}
	
	public double norm() {
		double res = 0;
		
		for(int i = 0; i < data.length; ++i) {
			res += data[i]*data[i];
		}
		
		return Math.sqrt(res);
	}
	
	public static Matrix eye(int n) {
		if(n < 1) {
			System.out.println("the param 'n' is under 1, which is invalid for func eye.");
			return null;
		}
		
		Matrix mtx = new Matrix(n, n);
				
		for(int i = 0; i < n; ++i) {
			for(int j = 0; j < n; ++j) {
				if(i == j) {
					mtx.data[i*n + j] = 1;
				} else {
					mtx.data[i*n + j] = 0;
				}
			}
		}
		
		return mtx;
	}

	public Matrix repmat(int mm, int nn) {
		if(mm < 1 || nn < 1) {
			System.out.println("the params 'mm' or 'nn' in repmat is invalid");
			return null;
		}
		
		Matrix res = new Matrix(m*mm, n*nn);
		
		int tx = n*nn;
		for(int i = 0; i < mm*m; ++i) {
			for(int j = 0; j < nn*n; ++j) {
				res.data[i*tx + j] = data[(i%m)*n + j%n];
			}
		}
		
/*		for(int i = 0; i < mm; ++i) {
			for(int j = 0; j < nn; ++j) {
				res = res.putValue(this, i*m, j*n);
				System.out.println("i*m:" + i*m + "  j*n:" + j*n);
			}
		}
*/		
		return res;
	}
	
	public Matrix mean() {
		if(n != 1) {
			Matrix res = new Matrix(1, n);

			for(int i = 0; i < n; ++i) {
				res.data[i] = 0;
				
				for(int j = 0; j < m; ++j) {
					res.data[i] += data[j*n + i];
				}
				
				res.data[i] /= m;
			}
			
			return res;
		} else {
			double res = 0;
			
			for(int i = 0; i < data.length; ++i) {
				res += data[i];
			}
			res /= data.length;
			
			return new Matrix(1, 1, res);
		}
	}

	public Matrix upper(double v) {
		Matrix res = new Matrix(this);
		
		for(int i = 0; i < data.length; ++i) {
			if(res.data[i] > v) {
				res.data[i] = 1;
			} else {
				res.data[i] = 0;
			}
		}
		
		return res;
	}
	
	public Matrix upper(Matrix mtx) {
		int mm = mtx.m, nn = mtx.n;
		
		if(mm != this.m || nn != this.n) {
			System.out.println("the size of the two matrix is not equal.");
			return null;
		} else {
			Matrix res = new Matrix(this);
			
			for(int i = 0; i < data.length; ++i) {
				if(res.data[i] > mtx.data[i]) {
					res.data[i] = 1;
				} else {
					res.data[i] = 0;
				}
			}
			
			return res;
		}
	}
	
	public boolean isContainNaN() {
		for(int i = 0; i < data.length; ++i) {
			if(Double.isNaN(data[i])) {
				return true;
			}
		}
		return false;
	}
	
	public boolean isContainInf() {
		for(int i = 0; i < data.length; ++i) {
			if(Double.isInfinite(data[i])) {
				return true;
			}
		}
		return false;
	}
	
	public Matrix reshape(int mm, int nn) {
		if(mm < 1 || nn < 1 || mm*nn != m*n) {
			System.out.println("mm or nn is invalid, or total total number is not equal to orignal matrix");
			return null;
		}
		
		Matrix res = new Matrix(mm, nn);
		
		int p, q, count = 0;
		for(int i = 0; i < nn; ++i) {
			for(int j = 0; j < mm; ++j) {
				q = count/m;
				p = count%m;
				
				res.data[j*nn + i] = data[p*n + q]; 
				count++;
			}
		}
		
		return res;
	}
	
	public static Matrix[] reshape(Matrix[] mtx, int mm, int nn, int cc) {
		int size = mtx[0].data.length;
		for(int i = 1; i < mtx.length; ++i) {
			if(size != mtx[i].data.length) {
				System.out.println("each dim of matrix[] is not equal.");
				return null;
			}
		}
		
		if(mm < 1 || nn < 1 || cc < 1 || mtx[0].m*mtx[0].n*mtx.length != mm*nn*cc) {
			System.out.println("mm or nn is invalid, or total total number is not equal to orignal matrix[].");
			return null;
		}
		
		Matrix[] res = new Matrix[cc];
		for(int i = 0; i < cc; ++i) {
			res[i] = new Matrix(mm, nn);
		}
		
		int m = mtx[0].m, n = mtx[0].n;
		Matrix tmp = mtx[0].reshape(1, m*n);
		for(int k = 1; k < mtx.length; ++k) {
			tmp = tmp.spliceInRowType(mtx[k].reshape(1, m*n));
		}
		
		int total = mm*nn;
		for(int i = 0; i < cc; ++i) {
			res[i] = tmp.subMatrix(0, total*i, 1, total, 0.0).trans().reshape(mm, nn);
		}
		
		return res;
	}
	
	public Matrix spliceInRowType(Matrix mtx) {
		if(m != mtx.m) {
			return null;
		}
		
		int nn = n+mtx.n, total = m*n;
		Matrix res = new Matrix(m, nn);
		
		int count = 0, rest;
		for(int i = 0; i < nn; ++i) {
			for(int j = 0; j < m; ++j) {
				if(count/total == 0) {
					res.data[j*nn + i] = data[j*n + i%n];
				} else {
					rest = count - total;
					res.data[j*nn + i] = mtx.data[j*mtx.n + (rest-j)/m];
				}
				count++;
			}
		}
		
		return res;
	}
	
	public Matrix nearestInterpolation(int h, int w) {
		Matrix res = Matrix.zeros(h, w);
		
		int idx = 0, idy = 0;
		double stepH = ((double)this.m) / h;
		double stepW = ((double)this.n) / w;
		
		for(int i = 0; i < h; ++i) {
			for(int j = 0; j < w; ++j) {
				idx = (int) (stepH*i);
				idy = (int) (stepW*j);
				
				res.data[i*w + j] = data[idx*this.n + idy];
			}
		}
		
		return res;
	}
	
	public Matrix spliceInRankType(Matrix mtx) {
		if(n != mtx.n) {
			return null;
		}
		
		int mm = m+mtx.m, total = m*n;
		Matrix res = new Matrix(mm, n);
		
		int count = 0, rest;
		for(int i = 0; i < mm; ++i) {
			for(int j = 0; j < n; ++j) {
				if(count/total == 0) {
					res.data[i*n + j] = data[i*n + j];
				} else {
					rest = count - total;
					res.data[i*n + j] = mtx.data[((rest-j)/n)*n + j];
				}
				count++;
			}
		}
		
		return res;
	}
	
	public static Matrix reshape(Matrix[] mtx, int mm, int nn) {
		int c = mtx.length;
		if(mm < 1 || nn < 1 || mm*nn != mtx[0].m*mtx[0].n*c) {
			System.out.println("mm or nn is invalid, or total total number is not equal to orignal matrix[].");
			return null;
		}
		
		int size = mtx[0].data.length;
		for(int i = 1; i < c; ++i) {
			if(size != mtx[i].data.length) {
				System.out.println("each dim of matrix[] is not equal.");
				return null;
			}
		}
				
		Matrix res = new Matrix(mm, nn);

		int h = mtx[0].m, w = mtx[0].n, total = h*w, count = 0;		
		int r, p, q, rest;
		for(int i = 0; i < nn; ++i) {
			for(int j = 0; j < mm; ++j) {
				r = count/total;
				rest = count%total;
				q = rest/h;
				p = rest%h;
				
				res.data[j*nn + i] = mtx[r].data[p*w + q];
				count++;
			}
		}
		
		return res;
	}
	
	public Matrix equal(Matrix mtx) {
		if((this.m == mtx.m) || (this.n == mtx.n)) {
			Matrix res = new Matrix(this);
			
			for(int i = 0; i < data.length; ++i) {
				if(data[i] == mtx.data[i]) {
					res.data[i] = 1.0;
				} else {
					res.data[i] = 0.0;
				}
			}
			
			return res;
		} else {
			System.out.println("the size of two matrix are not same.");
			return null;
		}
	}
	
	public Matrix trans() {
		Matrix res = new Matrix(n, m);
		
		for(int i = 0; i < n; ++i) {
			for(int j = 0; j < m; ++j) {
				res.data[i*m + j] = data[j*n + i];
			}
		}
		
		return res;
	}
	
	public static Matrix rand(int m, int n) {
		Matrix res = new Matrix(m, n);
		
		Random random = new Random();
		
		for(int i = 0; i < res.data.length; ++i) {
			res.data[i] = random.nextDouble();
		}
		
		return res;
	}
	
	public static Matrix randn(int m, int n) {
		Matrix res = new Matrix(m, n);
		
		Random random = new Random();
		
		for(int i = 0; i < res.data.length; ++i) {
			res.data[i] = random.nextGaussian();
		}
		
		return res;
	}
	
	public static void print(Matrix[] mtx) {
		for(int i = 0; i < mtx.length; ++i) {
			System.out.println("sclice[" + i + "] data:");
			mtx[i].print();
			System.out.println();
		}
	}
	
	public void print() {
		DecimalFormat  df = new DecimalFormat("####0.000");
		
		for(int i = 0; i < m; ++i) {
			for(int j = 0; j < n; ++j) {
				System.out.print(df.format(data[i*n + j]) + "\t");
			}
			System.out.println();
		}
	}
	
	public boolean putValue(double v, int i, int j) {
		if(!(i >= 0 && i < m) || !(j >= 0 && j < n)) {
			return false;
		}
		data[i*n + j] = v;
		return true;
	}
	
	public Matrix putValue(Matrix mtx, int x, int y) {
		int h = mtx.m, w = mtx.n;
		if(!(x >= 0 && x < m) || !(y >= 0 && y < n) || !(h > 0 && h <= m) || !(w > 0 && w <= n) || !(x+h <= m) || !(y+w <= n)) {
			System.out.println("x or y or h or w is invalid. putValue(Matrix, int, int)");
			return null;
		}
		
		Matrix res = new Matrix(this);
		for(int i = x; i < x+h; ++i) {
			for(int j = y; j < y+w; ++j) {
				res.data[i*n + j] = mtx.data[(i-x)*w + (j-y)];
			}
		}
		return res;
	}
	
	public double getValue(int i, int j) {
		if(i < 0 || i > m - 1 || j < 0 || j > n - 1) {
			return Math.tan(Math.PI/2); //NaN
		}
		return data[i*n + j];
	}
	
	public static Matrix[] subMatrix(Matrix[] mtx, int x, int y, int h, int w) {
		int m = mtx[0].m, n = mtx[0].n, c = mtx.length;
		if(!(x >= 0 && x < m) || !(y >= 0 && y < n) || !(h > 0 && h <= m) || !(w > 0 && w <= n)) {
			System.out.println("x or y or h or w is invalid. subMatrix(int, int, int, int, double)");
			return null;
		}

		int size = mtx[0].data.length;
		for(int i = 1; i < c; ++i) {
			if(size != mtx[i].data.length) {
				System.out.println("each dim of matrix[] is not equal.");
				return null;
			}
		}

		Matrix[] res = new Matrix[c];
		
		for(int i = 0; i < c; ++i) {
			res[i] = mtx[i].subMatrix(x, y, h, w, 0.0);
		}
		
		return res;
	}
	
	public Matrix subMatrix(int x, int y, int size, double fill) {
		if(!(x >= 0 && x < m) || !(y >= 0 && y < n) || !(size > 0 && size < ((m < n) ? m : n))) {
			System.out.println("x or y or size is invalid. subMatrix(int, int, int, double)");
			return null;
		}
		
		Matrix mtx = new Matrix(size, size);

		int half = size / 2;
		for(int i = 0; i < size*size; ++i) {
			mtx.data[i] = fill;
		}
		
		int spx = 0, spy = 0, epx = size, epy = size;
		int six = x - half, siy = y - half, eix = x + half + 1, eiy = y + half + 1;
		boolean xj = x < half, yj = y < half;
		
		if((x < half) || ((m - x) <= half)) {
			spx = xj ? (half - x) : 0;
			epx = xj ? size : (size - (x + half - m) - 1);
			
			six = xj ? 0 : (x - half);
			eix = xj ? (x + half + 1) : m;
		}
		if((y < half) || ((n - y) <= half)) {
			spy = yj ? (half - y) : 0;
			epy = yj ? size : (size - (y + half - n) - 1);
			
			siy = yj ? 0 : (y - half);
			eiy = yj ? (y + half + 1) : n;
		}

		for(int i = spx, p = six; i < epx && p < eix; ++i, ++p) {
			for(int j = spy, q = siy; j < epy && q < eiy; ++j, ++q) {
				mtx.data[i*size + j] = data[p*n + q];
			}
		}
		
		return mtx;
	}
	
	public Matrix subMatrix(int x, int y, int h, int w, double fill) {
		if(!(x >= 0 && x < m) || !(y >= 0 && y < n) || !(h > 0 && h <= m) || !(w > 0 && w <= n)) {
			System.out.println("x or y or h or w is invalid. subMatrix(int, int, int, int, double)");
			return null;
		}
		
		Matrix mtx = new Matrix(h, w);
		for(int i = 0; i < mtx.data.length; ++i) {
			mtx.data[i] = fill;
		}
		
		boolean xj = (m - x) < h, yj = (n - y) < w;
		int spx = 0, epx = h, spy = 0, epy = w;
		int six = x, eix = x + h + 1, siy = y, eiy = y + w + 1;
		
		epx = xj ? (h - (x + h - m)) : h;
		eix = xj ? m : (x + h + 1);
		epy = yj ? (w - (y + w - n)) : w;
		eiy = yj ? n : (y + w + 1);
		
		for(int i = spx, p = six; i < epx && p < eix; ++i, ++p) {
			for(int j = spy, q = siy; j < epy && q < eiy; ++j, ++q) {
				mtx.data[i*w + j] = data[p*n + q];
			}
		}
		
		return mtx;
	}

	public double[] getDataTypeDouble() {
		return data;
	}
	
	public int[] getDataTypeInt() {
		if(m != 0 && n != 0) {
			int[] res = new int[data.length];
			
			for(int i = 0; i < data.length; ++i) {
				res[i] = (int) data[i];
			}
			
			return res;
		} else {
			System.out.println("non data in this matrix.");
			return null;
		}
	}
	
	public int[] getSize() {
		int[] res = new int[2];

		res[0] = m;
		res[1] = n;
		
		return res;
	}
	
	public static Matrix[] meshgrid(int[] xgv, int[] ygv) {
		
		Matrix X = new Matrix(ygv.length, xgv.length);
		Matrix Y = new Matrix(ygv.length, xgv.length);
		
		for(int i = 0; i < ygv.length; ++i) {
			for(int j = 0; j < xgv.length; ++j) {
				X.data[i*xgv.length + j] = xgv[j];
				Y.data[i*xgv.length + j] = ygv[i];
			}
		}
		
		Matrix[] result = new Matrix[2];
		result[0] = X;
		result[1] = Y;
		
		return result;
	}	

	public Matrix sort() {
		Matrix res = this.reshape(m*n, 1);

		MatrixElemSort[] mes = new MatrixElemSort[m*n];
		int count = 0;
		for(int i = 0; i < m*n; ++i) {
			mes[i] = new MatrixElemSort(res.data[i], count);
			count ++;
		}
		java.util.Arrays.sort(mes);
		
		for(int i = 0; i < m*n; ++i) {
			res.data[i] = mes[i].loc;
		}
		
		return res;
	}
	
	public static Matrix[] mean(Matrix[] mtx) {
		int mm = mtx[0].m, nn = mtx[0].n, cc = mtx.length;
		
		Matrix[] res = new Matrix[cc];
		
		double val;
		for(int k = 0; k < cc; ++k) {
			res[k] = new Matrix(1, nn);

			for(int i = 0; i < nn; ++i) {
				val = 0;
				
				for(int j = 0; j < mm; ++j) {
					val += mtx[k].data[j*nn + i]; 
				}
				val /= mm;
				res[k].data[i] = val;
			}
		}
		
		return res;
	}
	
	public Matrix inv() {
		if(m != n) {
			System.out.println("m is not equal to n. inv()");
			return null;
		}
		
		Matrix res = new Matrix(this);
		
		int[] is = new int[n], js = new int[n];
		int l, u, v;
		double d, p;
		
		for(int k = 0; k < n; ++k) {
			d = 0;
			for(int i = k; i < n; ++i) {
				for(int j = k; j < n; ++j) {
					l = i*n + j;
					p = Math.abs(res.data[l]);
					
					if(p > d) {
						d = p;
						is[k] = i;
						js[k] = j;
					}
				}
			}
			if(d+1 == 1) {
				System.out.println(d);
				System.out.println("this matrix can not inverse.");
				return null;
			}
			if(is[k] != k) {
				for(int j = 0; j < n; ++j) {
					u = k*n + j;
					v = is[k]*n + j;
					
					p = res.data[u];
					res.data[u] = res.data[v];
					res.data[v] = p;
				}
			}
			if(js[k] != k) {
				for(int i = 0; i < n; ++i) {
					u = i*n + k;
					v = i*n + js[k];
					
					p = res.data[u];
					res.data[u] = res.data[v];
					res.data[v] = p;
				}
			}
			
			l = k*n + k;
			res.data[l] = 1.0/res.data[l];
			
			for(int j = 0; j < n; ++j) {
				if(j != k) {
					u = k*n + j;
					res.data[u] = res.data[u]*res.data[l];
				}
			}
			for(int i = 0; i < n; ++i) {
				if(i != k) {
					for(int j = 0; j < n; ++j) {
						if(j != k) {
							u = i*n + j;
							res.data[u] = res.data[u] - res.data[i*n+k]*res.data[k*n+j];
						}
					}
				}
			}
			for(int i = 0; i < n; ++i) {
				if(i != k) {
					u = i*n + k;
					res.data[u] = -res.data[u]*res.data[l];
				}
			}
		}
		
		for(int k = n-1; k >= 0; --k) {
			if(js[k] != k) {
				for(int j = 0; j < n; ++j) {
					u = k*n + j;
					v = js[k]*n + j;
					
					p = res.data[u];
					res.data[u] = res.data[v];
					res.data[v] = p;
				}
			}
			if(is[k] != k) {
				for(int i = 0; i < n; ++i) {
					u = i*n + k;
					v = i*n + is[k];
					
					p = res.data[u];
					res.data[u] = res.data[v];
					res.data[v] = p;
				}
			}
		}
		
		return res;
	}
	
	public Matrix cumSum(int rc) {
		Matrix res = new Matrix(this);
		
		if(rc == 1) {
			for(int i = 1; i < this.m; ++i) {
				for(int j = 0; j < this.n; ++j) {
					res.data[i*this.n + j] += res.data[(i-1)*this.n + j];
				}
			}
		} else if(rc == 2) {
			for(int i = 1; i < this.n; ++i) {
				for(int j = 0; j < this.m; ++j) {
					res.data[j*this.n + i] += res.data[j*this.n + i - 1];
				}
			}
		}
		
		return res;
	}
	
	public static void saveToFile(String filename, Matrix mtx) {
		if(filename != null && mtx != null) {
			try {
				FileOutputStream fos = new FileOutputStream(new File(filename));
				
				int param[] = new int[1];
				param[0] = mtx.m;
				fos.write(DataArrayToByteArray.convert(param), 0, 4);
				param[0] = mtx.n;
				fos.write(DataArrayToByteArray.convert(param), 0, 4);
				
				byte[] res = DataArrayToByteArray.convert(mtx.getDataTypeDouble());
				fos.write(res, 0, res.length);
				
				fos.close();
			} catch(Exception e) {
				e.printStackTrace();
			}
		}
	}
	
	public static Matrix loadFromFile(String filename) {
		if(filename != null) {
			try {
				FileInputStream fis = new FileInputStream(new File(filename));
				
				byte[] dat = new byte[4];
				fis.read(dat, 0, 4);
				int m = ByteToInteger.convert(dat, 4)[0];
				System.out.println("m = " + m);
				fis.read(dat, 0, 4);
				int n = ByteToInteger.convert(dat, 4)[0];
				System.out.println("n = " + n);
				
				byte[] datBuffer = new byte[m*n*8];
				fis.read(datBuffer, 0, m*n*8);
				
				Double[] mtxDatClass = (Double[])ByteArrayToDataArray.convert(datBuffer, new Double(0));
				double[] mtxDat = new double[m*n];
				for(int i = 0; i < m*n; ++i) {
					mtxDat[i] = (double)mtxDatClass[i];
				}
				Matrix res = new Matrix(mtxDat, m, n);

				fis.close();
				return res;
			} catch(Exception e) {
				e.printStackTrace();
			}
		} else {
			return null;
		}
		return null;
	}
	
	public static void main(String[] args) {
		final int size = 25;
		int[] data = new int[size];		
		for(int i = 0; i < size; ++i) {
			data[i] = i+1;
		}
		final int dx = 3;
		int[] dt = new int[dx];
		for(int i = 0; i < dx; ++i) {
			dt[i] = i+3;
		}
		
		int[] invTest = new int[size];
		invTest[0] = 2;		invTest[1] = 7;		invTest[2] = 1;
		invTest[3] = 2;		invTest[4] = 245;		invTest[5] = 3;
		invTest[6] = 6;		invTest[7] = 1;		invTest[8] = 6;
		
		
		Matrix o = new Matrix(invTest, 3, 3);
		o.print();
		System.out.println();
		o = o.div(255.0);
		o.print();
		System.out.println();
		Matrix pp = o.spliceInRowType(o.div(255.0));
		pp.print();
		System.out.println();
		
		o.equal(o.trans()).columnSum().print();
		
		
/*		o.print();
		Matrix.saveToFile("out.dat", o);
		System.out.println();
		Matrix i = Matrix.loadFromFile("out.dat");
		i.print();
*/		
		
//		Matrix t = new Matrix(data, 5, 5);
//		t.print();
//		System.out.println();
//		Matrix a = new Matrix(invTest, 3, 3);
//		a.print();
//		System.out.println();
//		a.repmat(2, 3).print();
//		t.putValue(a, 2, 2).print();
		
	//	Matrix b = Matrix.eye(3);
	//	a.columnAdd(2, 1, -1).print();
	//	a.cumSum(2).print();
//		b.print();
//		System.out.println("det:" + Matrix.eye(15).det().data[0]); 
	//	System.out.println();
		
//		Matrix[] res = Matrix.mean(Matrix.extendMatrixDim(a, 1));
//		res[0].print();
		
//		a.putValue(b, 1, 1);
//		a.print();
//		a.spliceInRowType(b).print();
		
/*
		Matrix win;
		for(int i = 0; i < 5; ++i) {
			for(int j = 0; j < 5; ++j) {
				win = a.subMatrix(i, j, 3, 0.0);
				a.print();
				win.print();
				System.out.println();
			}
		}
*/
//		Matrix[] c = Matrix.reshape(Matrix.extendMatrixDim(a, 3), 3, 3, 2);
//		Matrix.print(c);
//		Matrix b = a.reshape(1, size).subMatrix(0, 0, 1, 4, 0.0);
//		b.print();
//		System.out.println();
//		b.trans().reshape(2, 2).print();
	//	a.spliceInRankType(b).print();		
	}

}
