package com.utils.math;

public class MatrixElemSort implements Comparable {

	public double value;
	public int loc;
	
	public MatrixElemSort(double v, int l) {
		value = v;
		loc = l;
	}
	
	public static void main(String[] args) {
		MatrixElemSort[] o = new MatrixElemSort[4];
		
		for(int i = 0; i < 4; ++i) {
			o[i] = new MatrixElemSort(4-i, i+1);
		}
		
		java.util.Arrays.sort(o);
		
		for(int i = 0; i < 4; ++i) {
			System.out.println("v:" + o[i].value + "  loc:" + o[i].loc);
		}
	}

	public int compareTo(Object arg0) {
		MatrixElemSort v = (MatrixElemSort)arg0;
		if(this.value > v.value) {
			return 1;
		} else if(this.value < v.value) {
			return -1;
		} else {
			return 0;
		}
	}

}
