package com.autoencoder.utils.converter;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;

import com.utils.ByteToInteger;
import com.utils.math.Matrix;

public class MnistDataConverter extends DataConverter {

	public static final int CLASS_NUM = 10;
	
	private FileInputStream datInFile;	// the data file's stream.
	private FileInputStream lblInFile;	// the label file's stream.
	
	public MnistDataConverter() {
		this.totalRead = 0;
		
		this.datWidth = 0;
		this.datHeight = 0;
		
		this.datNumber = 0;
		this.lblNumber = 0;
		
		this.datInFile = null;
		this.lblInFile = null;
		
		this.data = null;
	}
	
	public MnistDataConverter(String datFile, String lblFile, int totalRead) {
		this.totalRead = totalRead;
		
		try {
			this.datInFile = new FileInputStream(new File(datFile));
			this.lblInFile = new FileInputStream(new File(lblFile));
		} catch(Exception e) {
			e.printStackTrace();
		}
	}
	
	@Override
	public void init() {
		if(this.lblInFile != null && this.datInFile != null) {
			try {
				// read image data.
				byte[] dat = new byte[4];
				this.datInFile.read(dat, 0, 4);	// read the magic number.

				this.datInFile.read(dat, 0, 4);	// read the total number of image.
				this.datNumber = ByteToInteger.convert(dat, 4)[0];	// convert byte array into integer.
//				System.out.println("imgNumber: " + datNumber);
				
				this.datInFile.read(dat, 0, 4);	// read the image's height;
				this.datHeight = ByteToInteger.convert(dat, 4)[0];
//				System.out.println("imgHeight: " + datHeight);

				this.datInFile.read(dat, 0, 4);	// read the image's width;
				this.datWidth = ByteToInteger.convert(dat, 4)[0];
//				System.out.println("imgWidth: " + datWidth);

				
				// read label data.
				this.lblInFile.read(dat, 0, 4);	// read the magic number.
				
				this.lblInFile.read(dat, 0, 4);	// read the total number of label.
				this.lblNumber = ByteToInteger.convert(dat, 4)[0];
//				System.out.println("lblNumber: " + lblNumber);
				
				if(this.datNumber != this.lblNumber) {
					throw new Exception("The number of image and label is not equal.");
				}
				
				// initialize the parameters.
				this.data = new ArrayList<ArrayList<Matrix>>();
				for(int i = 0; i < CLASS_NUM; ++i) {
					this.data.add(new ArrayList<Matrix>());
				}
			} catch(Exception e) {
				e.printStackTrace();
			}
		}
	}

	@Override
	public void read() {
		int imgBufferSize = this.datHeight*this.datWidth;
		int allImgBufferSize = this.datNumber*imgBufferSize;
		
		byte datBuffer[] = new byte[imgBufferSize];
		byte allImgDatBuffer[] = new byte[allImgBufferSize];
		byte lblBuffer[] = new byte[this.lblNumber];
		
		try {
			this.datInFile.read(allImgDatBuffer, 0, allImgBufferSize);
			this.lblInFile.read(lblBuffer, 0, this.lblNumber);
			
			// convert image: data array to matrix format.
			for(int n = 0; n < this.totalRead; ++n) {
				for(int i = 0; i < imgBufferSize; ++i) {
					datBuffer[i] = allImgDatBuffer[n*imgBufferSize + i];
				}

				Matrix mtx = new Matrix(ByteToInteger.convert(datBuffer, 1), this.datHeight, this.datWidth);
				int lbl = lblBuffer[n];
				
				this.data.get(lbl).add(mtx);
			}
			
//			for(int i = 0; i < MnistDataConverter.CLASS_NUM; ++i) {
//				System.out.println("Class " + i + " read " + this.data.get(i).size() + " images.");
//			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public void close() {
		try {
			if(this.datInFile != null) {
				this.datInFile.close();
			}
			
			if(this.lblInFile != null) {
				this.lblInFile.close();
			}
		} catch(Exception e) {
			e.printStackTrace();
		}
	}	
	
	public static int getClassNum() {
		return CLASS_NUM;
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		MnistDataConverter ic = new MnistDataConverter("./data/t10k-images-idx3-ubyte", "./data/t10k-labels-idx1-ubyte", 100);		// only read 100 images.
//		ImageConverter ic = new ImageConverter("./data/train-images.idx3-ubyte", "./data/train-labels.idx1-ubyte", 600);	// only read 600 images.
		ic.init();
		ic.read();
		ic.close();

		System.out.println("number of class: " + ic.getData().size());

	}

}
