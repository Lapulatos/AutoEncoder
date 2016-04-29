package com.autoencoder.utils;

import com.utils.math.Matrix;

public class MinimizeResult {
	
	private Matrix x;
	private Matrix fx;
	private double nIter;
	
	public MinimizeResult() {
		this.x = null;
		this.fx = null;
		this.nIter = 0;
	}
	
	public MinimizeResult(Matrix x, Matrix fx, double nIter) {
		this.x = x;
		this.fx = fx;
		this.nIter = nIter;
	}

	public Matrix getX() {
		return x;
	}

	public Matrix getFx() {
		return fx;
	}

	public double getnIter() {
		return nIter;
	}
	
}
