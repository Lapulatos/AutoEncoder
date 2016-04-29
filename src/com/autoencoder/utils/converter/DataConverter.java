package com.autoencoder.utils.converter;

import java.util.ArrayList;
import java.util.List;

import com.utils.math.Matrix;

public abstract class DataConverter {

	protected int totalRead;	// the data's total number that user want to read.
	
	protected int datWidth;		// the width of each data.
	protected int datHeight;		// the height of each data.
	
	protected int datNumber;		// the total number of data.
	protected int lblNumber;		// the total number of label.
	
	protected List<ArrayList<Matrix>> data;
	
	
	public abstract void init();
	public abstract void read();
	
	public int getTotalRead() {
		return totalRead;
	}

	public int getDatWidth() {
		return datWidth;
	}
	
	public int getDatHeight() {
		return datHeight;
	}
	
	public int getDatNumber() {
		return datNumber;
	}
	
	public int getLblNumber() {
		return lblNumber;
	}
	
	public List<ArrayList<Matrix>> getData() {
		return data;
	}
	
	
	
}
