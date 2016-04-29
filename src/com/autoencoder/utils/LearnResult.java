package com.autoencoder.utils;

import com.utils.math.Matrix;

public class LearnResult {

	private Matrix vishid;
	private Matrix hidbiases;
	private Matrix visbiases;
	
	public LearnResult() {
		this.vishid    = null;
		this.hidbiases = null;
		this.visbiases = null;
	}
	
	public LearnResult(Matrix vishid, Matrix hidbiases, Matrix visbiases) {
		this.vishid    = vishid;
		this.hidbiases = hidbiases;
		this.visbiases = visbiases;
	}
	
	public Matrix getVishid() {
		return vishid;
	}

	public Matrix getHidbiases() {
		return hidbiases;
	}

	public Matrix getVisbiases() {
		return visbiases;
	}

	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

}
