package com.autoencoder;

import java.util.ArrayList;
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
import com.utils.img.ProcessResultImageFrame;
import com.utils.math.Matrix;

public class BackPropagation {

	public static final int MIN_MINIMIZE_EPOCH = 5;
	
	private List<LearnResult> learnResult;
	private List<Batch> trainBatches;
	private List<Batch> testBatches;
	
	private int trainNumCases;
	private int trainNumDims;
	private int trainNumBatches;
	private int testNumCases;
	private int testNumDims;
	private int testNumBatches;
	
	private Matrix[] w;
	private int[] l;
	
	private Matrix testErr;
	private Matrix trainErr;
	
	public BackPropagation() {
		this.learnResult  = null;
		this.trainBatches = null;
		this.testBatches  = null;
		
		this.w = null;
		this.l = null;
		
		this.testErr  = null;
		this.trainErr = null;
	}
	
	public BackPropagation(ArrayList<LearnResult> learnResult, List<Batch> trainBatches, List<Batch> testBatches) {
		if((learnResult != null) && (learnResult.size() == 4) && (trainBatches != null) && (!trainBatches.isEmpty()) && (testBatches != null) && (!testBatches.isEmpty())) {
			this.learnResult  = learnResult;
			this.trainBatches = trainBatches;
			this.testBatches  = testBatches;
			
			
			// PREINITIALIZE WEIGHTS OF THE AUTOENCODER
			this.w = new Matrix[8];
			this.w[0] = this.learnResult.get(0).getVishid().spliceInRankType(this.learnResult.get(0).getHidbiases());
			this.w[1] = this.learnResult.get(1).getVishid().spliceInRankType(this.learnResult.get(1).getHidbiases());
			this.w[2] = this.learnResult.get(2).getVishid().spliceInRankType(this.learnResult.get(2).getHidbiases());
			this.w[3] = this.learnResult.get(3).getVishid().spliceInRankType(this.learnResult.get(3).getHidbiases());
			this.w[4] = this.learnResult.get(3).getVishid().trans().spliceInRankType(this.learnResult.get(3).getVisbiases());
			this.w[5] = this.learnResult.get(2).getVishid().trans().spliceInRankType(this.learnResult.get(2).getVisbiases());
			this.w[6] = this.learnResult.get(1).getVishid().trans().spliceInRankType(this.learnResult.get(1).getVisbiases());
			this.w[7] = this.learnResult.get(0).getVishid().trans().spliceInRankType(this.learnResult.get(0).getVisbiases());

			
			// END OF PREINITIALIZATIO OF WEIGHTS
			this.l = new int[9];
			this.l[0] = this.w[0].m - 1;
			this.l[1] = this.w[1].m - 1;
			this.l[2] = this.w[2].m - 1;
			this.l[3] = this.w[3].m - 1;
			this.l[4] = this.w[4].m - 1;
			this.l[5] = this.w[5].m - 1;
			this.l[6] = this.w[6].m - 1;
			this.l[7] = this.w[7].m - 1;
			this.l[8] = this.l[0];
			
			
			this.trainErr = Matrix.zeros(1, DeepAutoEncoder.MAX_EPOCH);
			this.testErr  = Matrix.zeros(1, DeepAutoEncoder.MAX_EPOCH);
		}
	}
	
	public void exec() throws Exception {
		ProcessResultImageFrame frame = new ProcessResultImageFrame();
		frame.setResizable(false);
		
		for(int i = 0; i < DeepAutoEncoder.MAX_EPOCH; ++i) {
			// COMPUTE TRAINING RECONSTRUCTION ERROR
			double err = 0;
			
			this.trainNumCases   = this.trainBatches.get(0).getNumCases();
			this.trainNumDims    = this.trainBatches.get(0).getNumDims();
			this.trainNumBatches = this.trainBatches.size();
			
			int h = 0, w = 0;
			Matrix data    = null;
			Matrix dataout = null;
			for(int j = 0; j < this.trainNumBatches; ++j) {
				Batch bh = this.trainBatches.get(j);
				data = bh.getBatchData();
				data = data.spliceInRowType(Matrix.ones(this.trainNumCases, 1));
				
				h = bh.getImgHeight();
				w = bh.getImgWidth();
				
				Matrix w1probs = Matrix.ones(data.m, this.w[0].n).dotDiv(data.multi(this.w[0]).multi(-1.0).exp().add(1.0)).spliceInRowType(Matrix.ones(this.trainNumCases, 1));
				Matrix w2probs = Matrix.ones(w1probs.m, this.w[1].n).dotDiv(w1probs.multi(this.w[1]).multi(-1.0).exp().add(1.0)).spliceInRowType(Matrix.ones(this.trainNumCases, 1));
				Matrix w3probs = Matrix.ones(w2probs.m, this.w[2].n).dotDiv(w2probs.multi(this.w[2]).multi(-1.0).exp().add(1.0)).spliceInRowType(Matrix.ones(this.trainNumCases, 1));
				Matrix w4probs = w3probs.multi(this.w[3]).spliceInRowType(Matrix.ones(this.trainNumCases, 1));
				Matrix w5probs = Matrix.ones(w4probs.m, this.w[4].n).dotDiv(w4probs.multi(this.w[4]).multi(-1.0).exp().add(1.0)).spliceInRowType(Matrix.ones(this.trainNumCases, 1));
				Matrix w6probs = Matrix.ones(w5probs.m, this.w[5].n).dotDiv(w5probs.multi(this.w[5]).multi(-1.0).exp().add(1.0)).spliceInRowType(Matrix.ones(this.trainNumCases, 1));
				Matrix w7probs = Matrix.ones(w6probs.m, this.w[6].n).dotDiv(w6probs.multi(this.w[6]).multi(-1.0).exp().add(1.0)).spliceInRowType(Matrix.ones(this.trainNumCases, 1));
				
				dataout = Matrix.ones(w6probs.m, this.w[7].n).dotDiv(w7probs.multi(this.w[7]).multi(-1.0).exp().add(1.0));
				err += (1.0 / this.trainNumCases) * (data.subMatrix(0, 0, data.m, (data.n - 1), 0.0).dotMinus(dataout).square().sum());
			}

			this.trainErr.putValue((err / this.trainNumBatches), 0, i);

			// END OF COMPUTING TRAINING RECONSTRUCTION ERROR
			
			
			// DISPLAY FIGURE TOP ROW REAL DATA BOTTOM ROW RECONSTRUCTIONS
			mnistDisp(frame, data.subMatrix(0, 0, 15, (data.n - 1), 0.0), dataout.subMatrix(0, 0, 15, (dataout.n), 0.0), h, w);
			
			// COMPUTE TEST RECONSTRUCTION ERROR
			err = 0;
			
			this.testNumCases   = this.testBatches.get(0).getNumCases();
			this.testNumDims    = this.testBatches.get(0).getNumDims();
			this.testNumBatches = this.testBatches.size();

			for(int j = 0; j < this.testNumBatches; ++j) {
				Batch bh = this.testBatches.get(j);
				data = bh.getBatchData();
				data = data.spliceInRowType(Matrix.ones(this.testNumCases, 1));
				
				Matrix w1probs = Matrix.ones(data.m, this.w[0].n).dotDiv(data.multi(this.w[0]).multi(-1.0).exp().add(1.0)).spliceInRowType(Matrix.ones(this.testNumCases, 1));
				Matrix w2probs = Matrix.ones(w1probs.m, this.w[1].n).dotDiv(w1probs.multi(this.w[1]).multi(-1.0).exp().add(1.0)).spliceInRowType(Matrix.ones(this.testNumCases, 1));
				Matrix w3probs = Matrix.ones(w2probs.m, this.w[2].n).dotDiv(w2probs.multi(this.w[2]).multi(-1.0).exp().add(1.0)).spliceInRowType(Matrix.ones(this.testNumCases, 1));
				Matrix w4probs = w3probs.multi(this.w[3]).spliceInRowType(Matrix.ones(this.testNumCases, 1));
				Matrix w5probs = Matrix.ones(w4probs.m, this.w[4].n).dotDiv(w4probs.multi(this.w[4]).multi(-1.0).exp().add(1.0)).spliceInRowType(Matrix.ones(this.testNumCases, 1));
				Matrix w6probs = Matrix.ones(w5probs.m, this.w[5].n).dotDiv(w5probs.multi(this.w[5]).multi(-1.0).exp().add(1.0)).spliceInRowType(Matrix.ones(this.testNumCases, 1));
				Matrix w7probs = Matrix.ones(w6probs.m, this.w[6].n).dotDiv(w6probs.multi(this.w[6]).multi(-1.0).exp().add(1.0)).spliceInRowType(Matrix.ones(this.testNumCases, 1));
				
				dataout = Matrix.ones(w6probs.m, this.w[7].n).dotDiv(w7probs.multi(this.w[7]).multi(-1.0).exp().add(1.0));
				err += (1.0 / this.testNumCases) * (data.subMatrix(0, 0, data.m, (data.n - 1), 0.0).dotMinus(dataout).square().sum());
			}

			this.testErr.putValue((err / this.testNumBatches), 0, i);

			System.out.printf("Before epoch %d Train squared error: %6.3f Test squared error: %6.3f \t \t \n", i, this.trainErr.getValue(0, i), this.testErr.getValue(0, i));
			
			// END OF COMPUTING TEST RECONSTRUCTION ERROR
			
			
			
			int tt = 0;
			for(int j = 0; j < (this.trainNumBatches / MIN_MINIMIZE_EPOCH); ++j) {
				System.out.printf("epoch %d batch %d\r", i, j);
				
				
				// COMBINE 10 MINIBATCHES INTO 1 LARGER MINIBATCH
				data = Matrix.ones(1, this.trainNumDims);	// empty matrix can not splice.
				for(int kk = 0; kk < MIN_MINIMIZE_EPOCH; ++kk) {
					data = data.spliceInRankType(this.trainBatches.get(tt*MIN_MINIMIZE_EPOCH + kk).getBatchData());
					System.out.printf("kk %d \r", kk);
				}
				data = data.subMatrix(1, 0, (data.m - 1), data.n, 0.0);	// discard the first row vector.
				tt  += 1; 
				
				double maxIter[] = new double[1];
				maxIter[0] = 3.0;
				
				Matrix VV = this.w[0].reshape((this.w[0].m * this.w[0].n), 1);
				for(int k = 1; k < 8; ++k) {
					VV = VV.spliceInRankType(this.w[k].reshape((this.w[k].m * this.w[k].n), 1));
				}
				Matrix dim = new Matrix(this.l, this.l.length, 1);
				
				
				Map<String, Matrix> params = new HashMap<String, Matrix>();
				params.put("dim", dim);
				params.put("xx", data);
				EnumMap<CGFunc, Map<String, Matrix>> args = new EnumMap<CGFunc, Map<String, Matrix>>(CGFunc.class);
				args.clear();
				args.put(CGFunc.CG_MNIST, params);
				
				MinimizeResult mRes = minimize(VV, CGFunc.CG_MNIST, maxIter, args);
				Matrix x  = mRes.getX();
				Matrix fx = mRes.getFx();
				
				
				this.w[0] = x.subMatrix(0, 0, ((this.l[0] + 1)*this.l[1]), 1, 0.0).reshape((this.l[0] + 1), this.l[1]);
				int xxx = (this.l[0] + 1)*this.l[1];
				this.w[1] = x.subMatrix(xxx, 0, ((this.l[1] + 1)*this.l[2]), 1, 0.0).reshape((this.l[1] + 1), this.l[2]);
				xxx += (this.l[1] + 1)*this.l[2];
				this.w[2] = x.subMatrix(xxx, 0, ((this.l[2] + 1)*this.l[3]), 1, 0.0).reshape((this.l[2] + 1), this.l[3]);
				xxx += (this.l[2] + 1)*this.l[3];
				this.w[3] = x.subMatrix(xxx, 0, ((this.l[3] + 1)*this.l[4]), 1, 0.0).reshape((this.l[3] + 1), this.l[4]);
				xxx += (this.l[3] + 1)*this.l[4];
				this.w[4] = x.subMatrix(xxx, 0, ((this.l[4] + 1)*this.l[5]), 1, 0.0).reshape((this.l[4] + 1), this.l[5]);
				xxx += (this.l[4] + 1)*this.l[5];
				this.w[5] = x.subMatrix(xxx, 0, ((this.l[5] + 1)*this.l[6]), 1, 0.0).reshape((this.l[5] + 1), this.l[6]);
				xxx += (this.l[5] + 1)*this.l[6];
				this.w[6] = x.subMatrix(xxx, 0, ((this.l[6] + 1)*this.l[7]), 1, 0.0).reshape((this.l[6] + 1), this.l[7]);
				xxx += (this.l[6] + 1)*this.l[7];
				this.w[7] = x.subMatrix(xxx, 0, ((this.l[7] + 1)*this.l[8]), 1, 0.0).reshape((this.l[7] + 1), this.l[8]);
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
					if((d3 > SIG*d0) || (f3 > (f0 + x3*RHO*d0)) || (m > 0)) {
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
	
	// every samples in realData or recData are column vector(m x 1), and only select five samples of each.
	public void mnistDisp(ProcessResultImageFrame frame, Matrix realData, Matrix recData, int h, int w) {		
		frame.setVisible(true);
		
		frame.setRealH(h);
		frame.setRealW(w);
		frame.setRecH(h);
		frame.setRecW(w);
		frame.setRealData(realData);
		frame.setRecData(recData);
		
		frame.paint();
	}
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

}
