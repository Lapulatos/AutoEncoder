package com.autoencoder;

import java.util.ArrayList;
import java.util.List;

import com.autoencoder.utils.Batch;
import com.autoencoder.utils.LearnResult;
import com.autoencoder.utils.converter.MnistDataConverter;
import com.utils.math.Matrix;

public class DeepAutoClassifier {

	public final static int MAX_EPOCH = 30;
	
	public static void main(String[] args) {
		int numHid = 500, numPen = 500, numPen2 = 2000;
		
		System.out.println("Converting Raw files into Matlab format.");
		
		MnistDataConverter icTrain = new MnistDataConverter("./data/train-images.idx3-ubyte", "./data/train-labels.idx1-ubyte", 600);	// only read 600 images.
		icTrain.init();
		icTrain.read();
		icTrain.close();
		MnistDataConverter icTeset = new MnistDataConverter("./data/t10k-images-idx3-ubyte", "./data/t10k-labels-idx1-ubyte", 100);	// only read 600 images.
		icTeset.init();
		icTeset.read();
		icTeset.close();		
		
		List<ArrayList<Matrix>> dataTrain = icTrain.getData();
		List<ArrayList<Matrix>> dataTest  = icTeset.getData();
		
		try {
			System.out.println("Pretraining a deep autoencoder.");
			System.out.printf("The Science paper used 50 epochs. This uses %3d\n", DeepAutoClassifier.MAX_EPOCH);
			List<Batch> trainBatches = Batch.makeBatch(dataTrain, 100);			
			List<Batch> testBatches = Batch.makeBatch(dataTest, 100);
			
			int numCases   = trainBatches.get(0).getNumCases();
			int numDims    = trainBatches.get(0).getNumDims();
			int numBatches = trainBatches.size();
			
			System.out.printf("Pretraining Layer 1 with RBM: %d-%d\n", numDims, numHid);
			RBM rbm1 = new RBM(trainBatches, numHid, true);
			List<Batch> l2baches = rbm1.exec();
			LearnResult mnistvhclassify = new LearnResult(rbm1.getVishid(), rbm1.getHidbiases(), rbm1.getVisbiases());
			
			System.out.printf("Pretraining Layer 2 with RBM: %d-%d\n", numHid, numPen);	
			RBM rbm2 = new RBM(l2baches, numPen, true);
			List<Batch> l3baches = rbm2.exec();
			LearnResult mnisthpclassify = new LearnResult(rbm2.getVishid(), rbm2.getHidbiases(), rbm2.getVisbiases());
			
			System.out.printf("Pretraining Layer 3 with RBM: %d-%d\n", numPen, numPen2);
			RBM rbm3 = new RBM(l3baches, numPen2, true);
			List<Batch> l4baches = rbm3.exec();
			LearnResult mnisthp2classify = new LearnResult(rbm3.getVishid(), rbm3.getHidbiases(), rbm3.getVisbiases());
						
			ArrayList<LearnResult> lAllLearnResult = new ArrayList<LearnResult>();
			lAllLearnResult.add(mnistvhclassify);
			lAllLearnResult.add(mnisthpclassify);
			lAllLearnResult.add(mnisthp2classify);
			
			BackPropagationClassify bpc = new BackPropagationClassify(lAllLearnResult, trainBatches, testBatches);
			bpc.exec();
		} catch(Exception e) {
			e.printStackTrace();
		}
	}

}
