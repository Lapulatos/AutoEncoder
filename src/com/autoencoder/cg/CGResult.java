package com.autoencoder.cg;

import com.utils.math.Matrix;

public class CGResult {

	private double f;
	private Matrix df;
	
	public CGResult() {
		this.f  = 0;
		this.df = null;
	}
	
	public CGResult(double f, Matrix df) {
		this.f  = f;
		this.df = df;
	}
	
	public double getF() {
		return f;
	}

	public Matrix getDf() {
		return df;
	}

}
