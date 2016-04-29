package com.autoencoder;

import java.util.ArrayList;
import java.util.List;

import com.autoencoder.utils.Batch;
import com.autoencoder.utils.LearnResult;
import com.autoencoder.utils.converter.ATandTDataConverter;
import com.autoencoder.utils.converter.MnistDataConverter;
import com.utils.math.Matrix;

public class DeepAutoEncoder {

	public final static int MAX_EPOCH = 10;
	
	
	public static void main(String[] args) {
		int numHid = 1000, numPen = 200, numPen2 = 100, numOpen = 16;
		
		System.out.println("Converting Raw files into Matlab format.");

		MnistDataConverter icTrain = new MnistDataConverter("./data/train-images.idx3-ubyte", "./data/train-labels.idx1-ubyte", 600);	// only read 600 images.
		icTrain.init();
		icTrain.read();
		icTrain.close();
		MnistDataConverter icTeset = new MnistDataConverter("./data/t10k-images-idx3-ubyte", "./data/t10k-labels-idx1-ubyte", 100);	// only read 600 images.
		icTeset.init();
		icTeset.read();
		icTeset.close();
		
//		List<ArrayList<Matrix>> dataTrain = attdc.getData();
//		List<ArrayList<Matrix>> dataTest  = attdc.getData();
		
		
		List<ArrayList<Matrix>> dataTrain = icTrain.getData();
		List<ArrayList<Matrix>> dataTest  = icTeset.getData();
		
		try {
			System.out.println("Pretraining a deep autoencoder.");
			List<Batch> trainBatches = Batch.makeBatch(dataTrain, 100);			
			List<Batch> testBatches = Batch.makeBatch(dataTest, 100);
			
			int numCases   = trainBatches.get(0).getNumCases();
			int numDims    = trainBatches.get(0).getNumDims();
			int numBatches = trainBatches.size();
			
			System.out.printf("Pretraining Layer 1 with RBM: %d-%d\n", numDims, numHid);
			RBM rbm1 = new RBM(trainBatches, numHid, true);
			List<Batch> l2baches = rbm1.exec();
			LearnResult l1LearnResult = new LearnResult(rbm1.getVishid(), rbm1.getHidbiases(), rbm1.getVisbiases());
			
			System.out.printf("Pretraining Layer 2 with RBM: %d-%d\n", numHid, numPen);	
			RBM rbm2 = new RBM(l2baches, numPen, true);
			List<Batch> l3baches = rbm2.exec();
			LearnResult l2LearnResult = new LearnResult(rbm2.getVishid(), rbm2.getHidbiases(), rbm2.getVisbiases());
			
			System.out.printf("Pretraining Layer 3 with RBM: %d-%d\n", numPen, numPen2);
			RBM rbm3 = new RBM(l3baches, numPen2, true);
			List<Batch> l4baches = rbm3.exec();
			LearnResult l3LearnResult = new LearnResult(rbm3.getVishid(), rbm3.getHidbiases(), rbm3.getVisbiases());
			
			System.out.printf("Pretraining Layer 4 with RBM: %d-%d\n", numPen2, numOpen);
			RBMHidLinear rbmhdl =  new RBMHidLinear(l4baches, numOpen, true);
			List<Batch> finalBaches =rbmhdl.exec();
			LearnResult l4LearnResult = new LearnResult(rbmhdl.getVishid(), rbmhdl.getHidbiases(), rbmhdl.getVisbiases());
			
			
			ArrayList<LearnResult> lAllLearnResult = new ArrayList<LearnResult>();
			lAllLearnResult.add(l1LearnResult);
			lAllLearnResult.add(l2LearnResult);
			lAllLearnResult.add(l3LearnResult);
			lAllLearnResult.add(l4LearnResult);
			
			BackPropagation bp = new BackPropagation(lAllLearnResult, trainBatches, testBatches);
			bp.exec();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

}
