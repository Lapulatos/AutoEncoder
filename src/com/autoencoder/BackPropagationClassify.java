package com.autoencoder;

import java.util.EnumMap;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.autoencoder.cg.CGResult;
import com.autoencoder.cg.ConjugateGradient;
import com.autoencoder.cg.ConjugateGradient.CGFunc;
import com.autoencoder.utils.Batch;
import com.autoencoder.utils.LearnResult;
import com.autoencoder.utils.MinimizeResult;
import com.utils.UtilFunction;
import com.utils.math.Matrix;

public class BackPropagationClassify {

	public static final int MIN_MINIMIZE_EPOCH = 5;

	private List<LearnResult> learnResult;
	private List<Batch> trainBatches;
	private List<Batch> testBatches;
	
	private int numTrainCases;
	private int numTrainDims;
	private int numTrainBatches;
	private int numTestCases;
	private int numTestDims;
	private int numTestBatches;
	
	private Matrix[] w;
	private int[]    l;
	private Matrix   wClass;
	
	private Matrix trainErr;
	private Matrix trainCrErr;
	private Matrix testErr;
	private Matrix testCrErr;
	
	public BackPropagationClassify() {
		this.learnResult  = null;
		this.trainBatches = null;
		this.testBatches  = null;
		
		this.numTrainBatches = 0;
		this.numTrainCases   = 0;
		this.numTrainDims    = 0;
		this.numTestBatches  = 0;
		this.numTestCases    = 0;
		this.numTestDims     = 0;
		
		this.w 		= null;
		this.l 		= null;
		this.wClass = null;
		
		this.trainErr   = null;
		this.trainCrErr = null;
		this.testErr    = null;
		this.testCrErr  = null;
	}
	
	public BackPropagationClassify(List<LearnResult> learnResult, List<Batch> trainBatches, List<Batch> testBatches) {
		if((learnResult != null) && (!learnResult.isEmpty()) && (learnResult.size() == 3) && (trainBatches != null) && (!trainBatches.isEmpty()) && (testBatches != null) && (!testBatches.isEmpty())) {
			this.learnResult  = learnResult;
			this.trainBatches = trainBatches;
			this.testBatches  = testBatches;
			
			this.numTrainBatches = this.trainBatches.size();
			this.numTrainCases   = this.trainBatches.get(0).getNumCases();
			this.numTrainDims    = this.trainBatches.get(0).getNumDims();
			
			this.numTestBatches = this.testBatches.size();
			this.numTestCases   = this.testBatches.get(0).getNumCases();
			this.numTestDims    = this.testBatches.get(0).getNumDims();
			
			this.w      = new Matrix[3];
			this.w[0]   = this.learnResult.get(0).getVishid().spliceInRankType(this.learnResult.get(0).getHidbiases());
			this.w[1]   = this.learnResult.get(1).getVishid().spliceInRankType(this.learnResult.get(1).getHidbiases());
			this.w[2]   = this.learnResult.get(2).getVishid().spliceInRankType(this.learnResult.get(2).getHidbiases());
			this.wClass = Matrix.randn((this.w[2].n + 1), 10).multi(0.1);
			
			this.l    = new int[5];
			this.l[0] = this.w[0].m - 1;
			this.l[1] = this.w[1].m - 1;
			this.l[2] = this.w[2].m - 1;
			this.l[3] = this.wClass.m - 1;
			this.l[4] = 10;
			
			this.trainErr   = Matrix.zeros(1, DeepAutoClassifier.MAX_EPOCH);
			this.trainCrErr = Matrix.zeros(1, DeepAutoClassifier.MAX_EPOCH);
			this.testErr    = Matrix.zeros(1, DeepAutoClassifier.MAX_EPOCH);
			this.testCrErr  = Matrix.zeros(1, DeepAutoClassifier.MAX_EPOCH);
		}
	}
	
	public void exec() throws Exception {
		// COMPUTE TRAINING MISCLASSIFICATION ERROR
		
		for(int i = 0; i < DeepAutoClassifier.MAX_EPOCH; ++i) {
			double err   = 0.0;
			double errCr = 0.0;
			int counter  = 0; 
			
			int n = this.numTrainCases;
			
			for(int j = 0; j < this.numTrainBatches; ++j) {
				Batch bh    = this.trainBatches.get(j);
				Matrix data = bh.getBatchData().spliceInRowType(Matrix.ones(n, 1));
				Matrix target = bh.getBatchLabel();
				
				Matrix w1probs = Matrix.ones(data.m, this.w[0].n).dotDiv(data.multi(this.w[0]).multi(-1.0).exp().add(1.0)).spliceInRowType(Matrix.ones(n, 1));
				Matrix w2probs = Matrix.ones(w1probs.m, this.w[1].n).dotDiv(w1probs.multi(this.w[1]).multi(-1.0).exp().add(1.0)).spliceInRowType(Matrix.ones(n, 1));
				Matrix w3probs = Matrix.ones(w2probs.m, this.w[2].n).dotDiv(w2probs.multi(this.w[2]).multi(-1.0).exp().add(1.0)).spliceInRowType(Matrix.ones(n, 1));
				
				Matrix targetOut = w3probs.multi(this.wClass).exp();
				targetOut = targetOut.dotDiv(targetOut.rowSum().repmat(1, 10));

				Matrix targetExpand = Matrix.zeros(targetOut.m, targetOut.n);
				for(int k = 0; k < target.m; ++k) {
					targetExpand.putValue(1.0, k, (int)target.data[k]);
				}
				
				Matrix[] ij   = targetOut.rowMax();
				Matrix[] i1j1 = targetExpand.rowMax();
				
				counter += ij[1].equal(i1j1[1]).columnSum().data[0];
				errCr   -= targetExpand.dotMulti(targetOut.log()).sum();
			}
			
			this.trainErr.data[i]   = this.numTrainCases*this.numTrainBatches - counter;
			this.trainCrErr.data[i] = errCr / (double)this.numTrainBatches;
			
			
			// END OF COMPUTING TRAINING MISCLASSIFICATION ERROR
			
			
			// COMPUTE TEST MISCLASSIFICATION ERROR
			err     = 0.0;
			errCr   = 0.0;
			counter = 0;
			
			n = this.numTestCases;
			
			for(int j = 0; j < this.numTestBatches; ++j) {
				Batch bh    = this.testBatches.get(j);
				Matrix data = bh.getBatchData().spliceInRowType(Matrix.ones(n, 1));
				Matrix target = bh.getBatchLabel();
				
				Matrix w1probs = Matrix.ones(data.m, this.w[0].n).dotDiv(data.multi(this.w[0]).multi(-1.0).exp().add(1.0)).spliceInRowType(Matrix.ones(n, 1));
				Matrix w2probs = Matrix.ones(w1probs.m, this.w[1].n).dotDiv(w1probs.multi(this.w[1]).multi(-1.0).exp().add(1.0)).spliceInRowType(Matrix.ones(n, 1));
				Matrix w3probs = Matrix.ones(w2probs.m, this.w[2].n).dotDiv(w2probs.multi(this.w[2]).multi(-1.0).exp().add(1.0)).spliceInRowType(Matrix.ones(n, 1));
				
				Matrix targetOut = w3probs.multi(this.wClass).exp();
				targetOut = targetOut.dotDiv(targetOut.rowSum().repmat(1, 10));
				
				Matrix targetExpand = Matrix.zeros(targetOut.m, targetOut.n);
				for(int k = 0; k < target.m; ++k) {
					targetExpand.putValue(1.0, k, (int)target.data[k]);
				}

				Matrix[] ij   = targetOut.rowMax();
				Matrix[] i1j1 = targetExpand.rowMax();
				
				counter += ij[1].equal(i1j1[1]).columnSum().data[0];
				errCr   -= targetExpand.dotMulti(targetOut.log()).sum();
			}
			
			this.testErr.data[i]   = this.numTestCases*this.numTestBatches - counter;
			this.testCrErr.data[i] = errCr / (double)this.numTestBatches;
			
			System.out.printf("Before epoch %d Train # misclassified: %d (from %d). Test # misclassified: %d (from %d) \t \t \n", i, (int)this.trainErr.data[i], this.numTrainCases*this.numTrainBatches, (int)this.testErr.data[i], this.numTestCases*this.numTestBatches);
			
			// END OF COMPUTING TEST MISCLASSIFICATION ERROR
			
			
			int tt = 0;
			for(int j = 0; j < (this.numTrainBatches / BackPropagationClassify.MIN_MINIMIZE_EPOCH); ++j) {
				System.out.printf("epoch %d batch %d\r", i, j);
				
				
				// COMBINE 10 MINIBATCHES INTO 1 LARGER MINIBATCH
				Matrix data    = Matrix.ones(1, this.numTrainDims);	// empty matrix can not splice.
				Matrix targets = Matrix.ones(1, 1);				// empty matrix can not splice.
				for(int kk = 0; kk < BackPropagationClassify.MIN_MINIMIZE_EPOCH; ++kk) {
					data = data.spliceInRankType(this.trainBatches.get(tt*BackPropagationClassify.MIN_MINIMIZE_EPOCH + kk).getBatchData());
					targets = targets.spliceInRankType(this.trainBatches.get(tt*BackPropagationClassify.MIN_MINIMIZE_EPOCH + kk).getBatchLabel());
				}
				data    = data.subMatrix(1, 0, (data.m - 1), data.n, 0.0);			// discard the first row vector.
				targets = targets.subMatrix(1, 0, (targets.m - 1), 1, 0.0);			// discard the first row vector.
				tt     += 1;
				
				
				// PERFORM CONJUGATE GRADIENT WITH 3 LINESEARCHES
				double maxIter[] = new double[1];
				maxIter[0] = 3.0;
				
				if(i < 5) {
					n = data.m;
					
					Matrix xx = data.spliceInRowType(Matrix.ones(n, 1));
					
					Matrix w1probs = Matrix.ones(xx.m, this.w[0].n).dotDiv(xx.multi(this.w[0]).multi(-1.0).exp().add(1.0)).spliceInRowType(Matrix.ones(n, 1));
					Matrix w2probs = Matrix.ones(w1probs.m, this.w[1].n).dotDiv(w1probs.multi(this.w[1]).multi(-1.0).exp().add(1.0)).spliceInRowType(Matrix.ones(n, 1));
					Matrix w3probs = Matrix.ones(w2probs.m, this.w[2].n).dotDiv(w2probs.multi(this.w[2]).multi(-1.0).exp().add(1.0));
					
					Matrix vv  = this.wClass.reshape(this.wClass.m*this.wClass.n, 1);
					Matrix dim = new Matrix(this.l, this.l.length, 1).subMatrix(3, 0, 2, 1, 0.0);
					
					Map<String, Matrix> params = new HashMap<String, Matrix>();
					params.put("dim", dim);
					params.put("w3probs", w3probs);
					params.put("target", targets);
					EnumMap<CGFunc, Map<String, Matrix>> args = new EnumMap<CGFunc, Map<String, Matrix>>(CGFunc.class);
					args.clear();
					args.put(CGFunc.CG_CLASSIFY_INIT, params);
					
					MinimizeResult mRes = minimize(vv, CGFunc.CG_CLASSIFY_INIT, maxIter, args);
					this.wClass = mRes.getX().reshape((this.l[3] + 1), this.l[4]);
				} else {
					Matrix vv  = this.w[0].reshape(this.w[0].m*this.w[0].n, 1).spliceInRankType(this.w[1].reshape(this.w[1].m*this.w[1].n, 1)).spliceInRankType(this.w[2].reshape(this.w[2].m*this.w[2].n, 1)).spliceInRankType(this.wClass.reshape(this.wClass.m*this.wClass.n, 1));
					Matrix dim = new Matrix(this.l, this.l.length, 1);
					
					Map<String, Matrix> params = new HashMap<String, Matrix>();
					params.put("dim", dim);
					params.put("xx", data);
					params.put("target", targets);
					EnumMap<CGFunc, Map<String, Matrix>> args = new EnumMap<CGFunc, Map<String, Matrix>>(CGFunc.class);
					args.clear();
					args.put(CGFunc.CG_CLASSIFY, params);
					
					MinimizeResult mRes = minimize(vv, CGFunc.CG_CLASSIFY, maxIter, args);
					Matrix x = mRes.getX();
					
					this.w[0] = x.subMatrix(0, 0, (this.l[0] + 1)*this.l[1], 1, 0.0).reshape((this.l[0] + 1), this.l[1]);
					int xxx   = (this.l[0] + 1)*this.l[1];
					this.w[1] = x.subMatrix(xxx, 0, (this.l[1] + 1)*this.l[2], 1, 0.0).reshape((this.l[1] + 1), this.l[2]);
					xxx += (this.l[1] + 1)*this.l[2];
					this.w[2] = x.subMatrix(xxx, 0, (this.l[2] + 1)*this.l[3], 1, 0.0).reshape((this.l[2] + 1), this.l[3]);
					xxx += (this.l[2] + 1)*this.l[3];
					this.wClass = x.subMatrix(xxx, 0, (this.l[3] + 1)*this.l[4], 1, 0.0).reshape((this.l[3] + 1), this.l[4]);
				}
				
				
				// END OF CONJUGATE GRADIENT WITH 3 LINESEARCHES
			}
		}
	}
	
	public MinimizeResult minimize(Matrix X,  CGFunc f, double[] len, EnumMap<CGFunc, Map<String, Matrix>> args) throws Exception {
		if(!args.isEmpty() && args.containsKey(f)) {
			Map<String, Matrix> params = args.get(f);
			params.put("vv", X);
			args.clear();
			args.put(f, params);
			
			final double INT = 0.1;
			final double EXT = 3.0;
			final int    MAX = 20;
			final double RATIO = 10;
			final double SIG   = 0.1;
			final double RHO   = SIG / 2;
			
			double red    = 0;
			double length = len[0]; 
			if(len.length == 2) {
				red = len[1];
				length = len[0];
			} else {
				red = 1.0;
			}
			
			String S = "";
			if(length > 0) {
				S = "Linesearch";
			} else {
				S = "Function evaluation";
			}
			
			double i = 0;
			boolean lsFailed = false;
			
			CGResult cgResult1 = ConjugateGradient.exec(f, args);
			double f0  = cgResult1.getF();
			Matrix df0 = cgResult1.getDf();
			Matrix fx = Matrix.ones(1, 1).multi(f0);
			
			i += (length < 0) ? 1 : 0;

			Matrix s  = df0.multi(-1.0);
			double d0 = s.trans().multi(s).multi(-1.0).data[0];
			double x2 = 0; double d2 = 0; double f2 = 0; 
			double x3 = red / (1.0 - d0); double d3 = 0; double f3 = 0;
			Matrix df3 = null;
			
			while(i < Math.abs(length)) {
				i += (length > 0) ? 1 : 0;
				
				Matrix X0  = new Matrix(X);
				double F0  = f0;
				Matrix dF0 = new Matrix(df0);
				
				double m = 0;
				if(length > 0) {
					m = MAX;
				} else {
					m = (MAX < (-length - i)) ? MAX : (-length - i);
				}
				
							
				while(true) {
					x2 = 0; f2 = f0; d2 = d0;
					f3 = f0;
					df3 = new Matrix(df0);
					
					boolean success = false;
					
					
					while(!success && m > 0) {
						try {
							m -= 1;
							i += (length < 0) ? 1 : 0;
							
							params.remove("vv");
							params.put("vv", X.dotAdd(s.multi(x3)));
							args.clear();
							args.put(f, params);
							
							CGResult cgResult3 = ConjugateGradient.exec(f, args);
							f3  = cgResult3.getF();
							df3 = cgResult3.getDf();
							
							if(Double.isNaN(f3) || Double.isInfinite(f3) || df3.isContainNaN() || df3.isContainInf()) {
								throw new Exception();
							}
							
							success = true;
						} catch(Exception e) {
							x3 = (x2 + x3) / 2.0;
						}
					}
					
					if(f3 < F0) {
						X0  = X.dotAdd(s.multi(x3));
						F0  = f3;
						dF0 = new Matrix(df3);
					}
					
					d3 = df3.trans().multi(s).data[0];
					if((d3 > SIG*d0) || (f3 > (f0 + x3*RHO*d0)) || (m == 0)) {
						break;
					}
					
					
					double x1 = x2; double f1 = f2; double d1 = d2;
					x2 = x3; f2 = f3; d2 = d3;
					
					double A = 6.0*(f1 - f2) + 3.0*(d2 + d1)*(x2 - x1);
					double B = 3.0*(f2 - f1) - (2.0*d1 + d2)*(x2 - x1);
					x3 = x1 - (d1*(x2 - x1)*(x2 - x1))/(B + Math.sqrt(B*B - A*d1*(x2 - x1)));
					
					if(Double.isNaN(x3) || Double.isInfinite(x3) || (x3 < 0)) {
						x3 = x2 * EXT;
					} else if(x3 > (x2 * EXT)) {
						x3 = x2*EXT;
					} else if(x3 < (x2 + INT*(x2 - x1))) {
						x3 = x2 + INT*(x2 - x1);
					}
				}
				
				
				while(((Math.abs(d3) > (-SIG*d0)) || (f3 > (f0 + x3*RHO*d0))) && (m > 0)) {
					double x4 = 0; double f4 = 0; double d4 = 0;
					
					if((d3 > 0) || (f3 > (f0 + x3*RHO*d0))) {
						x4 = x3; f4 = f3; d4 = d3;
					} else {
						x2 = x3; f2 = f3; d2 = d3;
					}
					
					if(f4 > f0) {
						x3 = x2 - (0.5*d2*(x4 - x2)*(x4 - x2))/(f4 - f2 - d2*(x4 - x2));
					} else {
						double A = 6.0*(f2 - f4)/(x4 - x2) + 3.0*(d4 + d2);
						double B = 3.0*(f4 - f2) - (2.0*d2 + d4)*(x4 - x2);
						x3 = x2 + (Math.sqrt(B*B - A*d2*(x4 - x2)*(x4 - x2)) - B) / A;
					}
					
					if(Double.isNaN(x3) || Double.isInfinite(x3)) {
						x3 = (x2 + x4) / 2;
					}
					
					x3 = UtilFunction.max(UtilFunction.min(x3, (x4 - INT*(x4 - x2))), (x2 + INT*(x4 - x2)));
					
					params.remove("vv");
					params.put("vv", X.dotAdd(s.multi(x3)));
					args.clear();
					args.put(f, params);
					
					CGResult cgResult3 = ConjugateGradient.exec(f, args);
					f3  = cgResult3.getF();
					df3 = cgResult3.getDf();
					
					if(f3 < F0) {
						X0  = X.dotAdd(s.multi(x3));
						F0  = f3;
						dF0 = new Matrix(df3);
					}
					
					m -= 1;
					i += (length < 0) ? 1 : 0;
					d3 = df3.trans().multi(s).data[0];
				}
				
				
				if((Math.abs(d3) < (-SIG*d0)) && (f3 < (f0 + x3*RHO*d0))) {
					X  = X.dotAdd(s.multi(x3));
					f0 = f3; 
					fx = fx.spliceInRankType(Matrix.ones(1, 1).multi(f0));
					
					System.out.printf("%s %6.4f;  Value %4.6e\r", S, i, f0);
					
					
					s = s.multi((df3.trans().multi(df3).data[0] - df0.trans().multi(df3).data[0]) / (df0.trans().multi(df0).data[0])).dotMinus(df3);
					df0 = new Matrix(df3);
					d3 = d0; d0 = df0.trans().multi(s).data[0];
					
					if(d0 > 0) {
						s = df0.multi(-1.0);
						d0 = s.trans().multi(s).multi(-1.0).data[0];
					}
					
					x3 = x3 * UtilFunction.min(RATIO, (d3 / (d0 - Double.MIN_NORMAL)));
					lsFailed = false;
				} else {
					X  = new Matrix(X0);
					f0 = F0;
					df0 = new Matrix(dF0);
					
					if(lsFailed || (i > Math.abs(length))) {
						break;
					}
					
					s = new Matrix(df0.multi(-1.0));
					d0 = s.trans().multi(s).multi(-1.0).data[0];
					x3 = 1.0 / (1.0 - d0);
					lsFailed = true;
				}
			}
			
			System.out.println();
			
			MinimizeResult mRes = new MinimizeResult(X, fx, i);
			return mRes;
		}
		
		return null;
	}
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

}
