package com.autoencoder.utils;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import com.utils.ByteToInteger;
import com.utils.math.Matrix;

public class ImageConverter {
	
	private int totalRead;	// the image's total number that user want to read.
	
	private int imgWidth;	// the width of each image.
	private int imgHeight;	// the height of each image.
	
	private int imgNumber;	// the total number of image.
	private int lblNumber;	// the total number of label.
	
	public final int CLASS_NUM = 10;	// the number of different class.
	
	public List<ArrayList<Matrix>> imgData;	// storage all the images into here.
	
	private FileInputStream datInFile;	// the data file's stream.
	private FileInputStream lblInFile;	// the label file's stream.
	
	public ImageConverter() {
		this.totalRead = 0;
		
		this.imgWidth = 0;
		this.imgHeight = 0;

		this.imgNumber = 0;
		this.lblNumber = 0;
		
		this.imgData = null;
		datInFile = null;
		lblInFile = null;
	}

	public ImageConverter(String datFile, String lblFile, int totalRead) {
		this.totalRead = totalRead;
		
		try {
			this.datInFile = new FileInputStream(new File(datFile));
			this.lblInFile = new FileInputStream(new File(lblFile));

			
			// read image data.
			byte[] dat = new byte[4];
			datInFile.read(dat, 0, 4);	// read the magic number.

			datInFile.read(dat, 0, 4);	// read the total number of image.
			this.imgNumber = ByteToInteger.convert(dat, 4)[0];	// convert byte array into integer.
//			System.out.println("imgNumber: " + imgNumber);
			
			datInFile.read(dat, 0, 4);	// read the image's height;
			this.imgHeight = ByteToInteger.convert(dat, 4)[0];
//			System.out.println("imgHeight: " + imgHeight);

			datInFile.read(dat, 0, 4);	// read the image's width;
			this.imgWidth = ByteToInteger.convert(dat, 4)[0];
//			System.out.println("imgWidth: " + imgWidth);

			
			// read label data.
			lblInFile.read(dat, 0, 4);	// read the magic number.
			
			lblInFile.read(dat, 0, 4);	// read the total number of label.
			this.lblNumber = ByteToInteger.convert(dat, 4)[0];
//			System.out.println("lblNumber: " + lblNumber);
			
			if(this.imgNumber != this.lblNumber) {
				throw new Exception("The number of image and label is not equal.");
			}
			
			// initialize the parameters.
			this.imgData = new ArrayList<ArrayList<Matrix>>();
			for(int i = 0; i < CLASS_NUM; ++i) {
				this.imgData.add(new ArrayList<Matrix>());
			}
		} catch(Exception e) {
			e.printStackTrace();
		}
	}
	
	public void read() {
		int imgBufferSize = this.imgHeight*this.imgWidth;
		int allImgBufferSize = this.imgNumber*imgBufferSize;
		
		byte datBuffer[] = new byte[imgBufferSize];
		byte allImgDatBuffer[] = new byte[allImgBufferSize];
		byte lblBuffer[] = new byte[this.lblNumber];
		
		try {
			this.datInFile.read(allImgDatBuffer, 0, allImgBufferSize);
			this.lblInFile.read(lblBuffer, 0, this.lblNumber);
			
			// convert image: data array to matrix format.
			for(int n = 0; n < totalRead; ++n) {
				for(int i = 0; i < imgBufferSize; ++i) {
					datBuffer[i] = allImgDatBuffer[n*imgBufferSize + i];
				}

				Matrix mtx = new Matrix(ByteToInteger.convert(datBuffer, 1), this.imgHeight, this.imgWidth);
				int lbl = lblBuffer[n];
//				System.out.println("lbl: " + lbl);
				
				this.imgData.get(lbl).add(mtx);
			}
			
//			for(int i = 0; i < this.classNum; ++i) {
//				System.out.println("Class " + i + " read " + this.imgData.get(i).size() + " images.");
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
	
	public int getImgWidth() {
		return imgWidth;
	}

	public int getImgHeight() {
		return imgHeight;
	}

	public int getImgNumber() {
		return imgNumber;
	}

	public int getLblNumber() {
		return lblNumber;
	}

	public List<ArrayList<Matrix>> getImgData() {
		return imgData;
	}

	public int getClassNum() {
		return CLASS_NUM;
	}

	public static void main(String[] args) {
		ImageConverter ic = new ImageConverter("./data/t10k-images-idx3-ubyte", "./data/t10k-labels-idx1-ubyte", 100);		// only read 100 images.
//		ImageConverter ic = new ImageConverter("./data/train-images.idx3-ubyte", "./data/train-labels.idx1-ubyte", 600);	// only read 600 images.
		ic.read();
		ic.close();

		System.out.println("number of class: " + ic.getImgData().size());
//		Matrix mtx = ic.getImgData().get(0).get(0);
//		mtx.print();
	}

}
