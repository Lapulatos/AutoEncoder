package com.autoencoder;

import java.util.ArrayList;
import java.util.List;

import com.autoencoder.utils.Batch;
import com.utils.math.Matrix;

public class RBM {

	// const parameter of this class.
	public final static double EPSILON_W  = 0.1;
	public final static double EPSILON_VB = 0.1;
	public final static double EPSILON_HB = 0.1;
	
	public final static double WEIGHT_COST      = 0.0002;
	public final static double INITIAL_MOMENTUM = 0.5;
	public final static double FINAL_MOMENTUM   = 0.9;
		

	private int     epoch;
	private boolean isRestart = true;
	
	private Matrix vishid;
	private Matrix hidbiases;
	private Matrix visbiases;

	private Matrix poshidprobs;
	private Matrix neghidprobs;
	private Matrix posprods;
	private Matrix negprods;
	private Matrix vishidinc;
	private Matrix hidbiasinc;
	private Matrix visbiasinc;
	private Matrix[] batchposhidprobs;
	
	private int numCases;
	private int numBatches;
	private int numDims;
	private int numHid;
	
	private List<Batch> batches;
	
	public RBM() {
		this.vishid = null;
		this.hidbiases = null;
		this.visbiases = null;

		this.poshidprobs = null;
		this.neghidprobs = null;
		this.posprods = null;
		this.negprods = null;
		this.vishidinc = null;
		this.hidbiasinc = null;
		this.visbiasinc = null;
		this.batchposhidprobs = null;
		
		this.numCases = 0;
		this.numBatches = 0;
		this.numDims = 0;
		this.numHid = 0;
	}
	
	public RBM(List<Batch> batches, int numHid, boolean isRestart) {
		if(batches != null && !batches.isEmpty() && isRestart) {
			this.isRestart = false;
			this.epoch     = 1;
			this.batches   = batches;
			
			this.numCases   = batches.get(0).getNumCases();
			this.numDims    = batches.get(0).getNumDims();
			this.numBatches = batches.size(); 
			this.numHid     = numHid;

			
			this.vishid    = Matrix.randn(numDims, numHid).multi(0.1);
			this.hidbiases = Matrix.zeros(1, numHid);
			this.visbiases = Matrix.zeros(1, numDims);

			this.poshidprobs = Matrix.zeros(numCases, numHid);
			this.neghidprobs = Matrix.zeros(numCases, numHid);
			this.posprods    = Matrix.zeros(numDims, numHid);
			this.negprods    = Matrix.zeros(numDims, numHid);
			this.vishidinc   = Matrix.zeros(numDims, numHid);
			this.hidbiasinc  = Matrix.zeros(1, numHid);
			this.visbiasinc  = Matrix.zeros(1, numDims);
			
			this.batchposhidprobs = new Matrix[numBatches];
			for(int i = 0; i < numBatches; ++i) {
				this.batchposhidprobs[i] = Matrix.zeros(numCases, numHid);
			}
		}
	}
	
	public List<Batch> exec() throws Exception {
		for(int i = 0; i < DeepAutoEncoder.MAX_EPOCH; ++i) {
			double errSum = 0;
			
			for(int j = 0; j < this.numBatches; ++j) {
				// START POSITIVE PHASE
				Batch  bh = this.batches.get(j);
				Matrix data = bh.getBatchData();				
				
				this.poshidprobs = Matrix.ones(numCases, numHid).dotDiv(data.multi(this.vishid).multi(-1.0).minus(this.hidbiases.repmat(this.numCases, 1)).exp().add(1.0));
				this.batchposhidprobs[j] = this.poshidprobs;
				this.posprods = data.trans().multi(this.poshidprobs);
				
				Matrix poshidact = this.poshidprobs.columnSum();
				Matrix posvisact = data.columnSum();
				
				
				// END OF POSITIVE PHASE
				Matrix poshidstates = this.poshidprobs.upper(Matrix.rand(this.numCases, this.numHid));
				
				
				// START NEGATIVE PHASE
				Matrix negdata = Matrix.ones(this.numCases, this.numDims).dotDiv(poshidstates.multi(this.vishid.trans()).multi(-1.0).minus(this.visbiases.repmat(this.numCases, 1)).exp().add(1.0));
				this.neghidprobs = Matrix.ones(this.numCases, this.numHid).dotDiv(negdata.multi(this.vishid).multi(-1.0).minus(this.hidbiases.repmat(this.numCases, 1)).exp().add(1.0));

				this.negprods = negdata.trans().multi(neghidprobs);
				Matrix neghidact = neghidprobs.columnSum();
				Matrix negvisact = negdata.columnSum();
				
				
				// END OF NEGATIVE PHASE
				double err = data.minus(negdata).square().sum();
				System.out.printf(" err %6.1f  \n", err);
				errSum += err;
				System.out.printf(" errSum %6.1f  \n", errSum);
				
				double momentum;
				if(epoch > 5) {
					momentum = RBM.FINAL_MOMENTUM;
				} else {
					momentum = RBM.INITIAL_MOMENTUM;
				}
				
				
				// UPDATE WEIGHTS AND BIASES
				this.vishidinc  = this.vishidinc.multi(momentum).dotAdd(this.posprods.dotMinus(this.negprods).div(this.numCases).minus(this.vishid.multi(RBM.WEIGHT_COST)).multi(RBM.EPSILON_W));
				this.visbiasinc = this.visbiasinc.multi(momentum).dotAdd(posvisact.dotMinus(negvisact).multi((RBM.EPSILON_VB / this.numCases))); 
				this.hidbiasinc = this.hidbiasinc.multi(momentum).dotAdd(poshidact.dotMinus(neghidact).multi((RBM.EPSILON_HB / this.numCases)));
				
				this.vishid    = this.vishid.dotAdd(this.vishidinc);
				this.visbiases = this.visbiases.dotAdd(this.visbiasinc);
				this.hidbiases = this.hidbiases.dotAdd(this.hidbiasinc);
				
				
				//END OF UPDATES
			}
			
			System.out.printf("epoch %4d error %6.1f  \n", i, errSum);
		}
		
		ArrayList<Batch> resBatch = new ArrayList<Batch>();
		for(int i = 0; i < this.batchposhidprobs.length; ++i) {
			Batch newBatch= new Batch(this.batchposhidprobs[i], this.batches.get(i).getImgHeight(), this.batches.get(i).getImgWidth(), this.batches.get(i).getBatchLabel());
			resBatch.add(newBatch);
		}
		
		return resBatch;
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

	public Matrix[] getBatchposhidprobs() {
		return batchposhidprobs;
	}

	public static void main(String[] args) {
		
	}

}
