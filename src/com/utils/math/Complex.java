package com.utils.math;

public class Complex {

	public double real;
	public double imag;
	
	public Complex() {
		this.real = 0.0;
		this.imag = 0.0;
	}
	
	public Complex(double real, double imag) {
		this.real = real;
		this.imag = imag;
	}
	
	public Complex(Complex cpx) {
		this.real = cpx.real;
		this.imag = cpx.imag;
	}
	
	public Complex add(double v) {
		Complex res = new Complex(this);
		
		res.real += v;
		
		return res;
	}
	
	public Complex add(Complex cpx) {
		Complex res = new Complex(this);
		
		res.real += cpx.real;
		res.imag += cpx.imag;
		
		return res;
	}
	
	public Complex minus(double v) {
		Complex res = new Complex(this);
		
		res.real -= v;
		
		return res;
	}
	
	public Complex minus(Complex cpx) {
		Complex res = new Complex(this);
		
		res.real -= cpx.real;
		res.imag -= cpx.imag;
		
		return res;
	}
	
	public Complex multi(double v) {
		Complex res = new Complex(this);
		
		res.real *= v;
		res.imag *= v;
		
		return res;
	}
	
	public Complex multi(Complex cpx) {
		Complex res = new Complex();
		
		res.real = (this.real*cpx.real - this.imag*cpx.imag);
		res.imag = (this.imag*cpx.real - this.real*cpx.imag);
		
		return res;
	}
	
	public Complex div(double v) {
		Complex res = new Complex();
		
		res.real = (this.real*v) / (v*v);
		res.imag = (this.imag*v) / (v*v);
		
		return res;
	}
	
	public Complex div(Complex cpx) {
		Complex res = new Complex();
		
		res.real = (this.real*cpx.real + this.imag*cpx.imag) / (cpx.real*cpx.real + cpx.imag*cpx.imag);
		res.imag = (this.imag*cpx.real - this.real*cpx.imag) / (cpx.real*cpx.real + cpx.imag*cpx.imag);
		
		return res;
	}
	
	public boolean isContainImag() {
		if(this.imag != 0) {
			return true;
		} else {
			return false;
		}
	}
	
	public void print() {
		if(this.imag >= 0) {
			System.out.printf("%6.4f + %6.4fi\n", this.real, this.imag);
		} else {
			System.out.printf("%6.4f - %6.4fi\n", this.real, Math.abs(this.imag));
		}
	}
	
	public static void main(String[] args) {
		Complex cpx1 = new Complex(1.5, 2.3);
		Complex cpx2 = new Complex(-1.7, 3.1);
		cpx1.print();
		cpx2.print();
		cpx1.div(cpx2).print();
	}

}
