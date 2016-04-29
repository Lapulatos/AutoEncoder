package com.autoencoder.utils;

import java.util.ArrayList;
import java.util.List;

import com.utils.math.Matrix;

public class Batch {

	private int numCases;
	private int numDims;
	
	private int imgHeight;
	private int imgWidth;
	
	private Matrix batchData;	// contain m samples, each sample is a n dimension vector( 1 x n).
	private Matrix batchLabel;	// batchLabel is a vector witch have m sample(m x 1).
	
	public Batch() {
		this.numCases = 0;
		this.numDims = 0;
		
		this.imgHeight = 0;
		this.imgWidth = 0;
		
		this.batchData = null;
		this.batchLabel = null;
	}
	
	public Batch(Matrix data, int height, int width, Matrix label) throws Exception {
		if(data.m != label.m) {
			throw new Exception("The number of data sample is not equal to the label's.");
		} else {
			this.numCases = data.m;
			this.numDims  = data.n;
			
			this.imgHeight = height;
			this.imgWidth  = width;
			
			this.batchData = new Matrix(data);
			this.batchLabel = new Matrix(label);
		}
	}
		
	public static List<Batch> makeBatch(List<ArrayList<Matrix>> data, int nSamplePerBatch) throws Exception {
		ArrayList<Batch> res = new ArrayList<Batch>();

		if(data != null && !data.isEmpty() && !data.get(0).isEmpty()) {
			Matrix tmp = data.get(0).get(0);
			int h = tmp.m, w = tmp.n;
			int mtxSize = h * w;
			int nClass = data.size();

			int nPerClass[] = new int[nClass];
			int nSample = 0;
			for(int i = 0; i < nClass; ++i) {
				nPerClass[i] = data.get(i).size();
				nSample += nPerClass[i];
			}
			
			if(nSample % nSamplePerBatch == 0) {
				// put all the data into datMtx.
				Matrix datMtx = new Matrix(nSample, mtxSize);

				int count = 0;
				for(int i = 0; i < nClass; ++i) {
					for(int j = 0; j < nPerClass[i]; ++j) {
						tmp = data.get(i).get(j).reshape(1, mtxSize);
						datMtx = datMtx.putValue(tmp, count++, 0);
					}
				}
				datMtx = datMtx.div(255.0);			// scale the element of data matrix(0 ~ 1).
				
				// generate the label matrix.
				Matrix lblMtx = new Matrix(nSample, 1);
				count = 0;
				for(int i = 0; i < nClass; ++i) {
					for(int j = 0; j < nPerClass[i]; ++j) {
						lblMtx.putValue(i, count++, 0);
					}
				}
				
				int[] randSample = Batch.randomCommon(nSample);
				int   nBatch     = nSample / nSamplePerBatch;
				
				for(int i = 0; i < nBatch; ++i) {
					Matrix subDatMtx = new Matrix(nSamplePerBatch, mtxSize);
					Matrix subLblMtx = new Matrix(nSamplePerBatch, 1);

					for(int j = 0; j < nSamplePerBatch; ++j) {
						subDatMtx = subDatMtx.putValue(datMtx.subMatrix(randSample[i*nSamplePerBatch + j], 0, 1, mtxSize, 0.0), j, 0);
						subLblMtx.putValue(lblMtx.getValue(randSample[i*nSamplePerBatch + j], 0), j, 0);
					}
					
					res.add(new Batch(subDatMtx, h, w, subLblMtx));
				}
				
			} else {
				throw new Exception("Can not divide all the samples to every Batches that contain same number of samples.");
			}
		}
		
		return res;
	}
	
	public static int[] randomCommon(int n) {
		int result[] = new int[n];
		for(int i = 0; i < n; ++i) {
			result[i] = -1;
		}
		
		for(int i = 0; i < n; ++i) {
			int idx = (int)(Math.random()*(n - 1));
			
			if(result[idx] == -1) {
				result[idx] = i;
			} else {
				while(result[idx] != -1) {
					idx = (idx >= (n - 1)) ? 0 : (idx + 1);
				}
				result[idx] = i;
			}
		}
		return result;
	}
	
	public Matrix getBatchData() {
		return batchData;
	}

	public Matrix getBatchLabel() {
		return batchLabel;
	}
	
	public int getNumCases() {
		return numCases;
	}

	public int getNumDims() {
		return numDims;
	}

	public int getImgHeight() {
		return imgHeight;
	}

	public int getImgWidth() {
		return imgWidth;
	}

	public static void main(String[] args) throws Exception {
		ImageConverter ic = new ImageConverter("./data/train-images.idx3-ubyte", "./data/train-labels.idx1-ubyte", 600);	// only read 600 images.
		ic.read();
		ic.close();
		
		ArrayList<Batch> res = (ArrayList<Batch>) Batch.makeBatch(ic.getImgData(), 100);
		System.out.println("Batch Number:" + res.size());

		res.get(0).getBatchData().subMatrix(0, 0, 1, 784, 0).reshape(28, 28).print();
		System.out.println("Label: " + res.get(0).getBatchLabel().getValue(0, 0));
//		res.get(0).getBatchLabel().print();
	}

}
