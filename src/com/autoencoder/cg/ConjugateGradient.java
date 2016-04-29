package com.autoencoder.cg;

import java.util.EnumMap;
import java.util.Map;

import com.utils.math.Matrix;

public class ConjugateGradient {
	
	public static enum CGFunc {
		CG_CLASSIFY,
		CG_CLASSIFY_INIT,
		CG_MNIST;
	}
	
	private static CGResult cgClassfiy(Matrix vv, Matrix dim, Matrix xx, Matrix target) {
		int l1 = (int) dim.data[0];
		int l2 = (int) dim.data[1];
		int l3 = (int) dim.data[2];
		int l4 = (int) dim.data[3];
		int l5 = (int) dim.data[4];
		int n  = xx.m;
		
		
		// Do decomversion.
		Matrix w1 = vv.subMatrix(0, 0, ((l1 + 1)*l2), 1, 0.0).reshape(l1+1, l2);
		int xxx = (l1 + 1)*l2;
		Matrix w2 = vv.subMatrix(xxx, 0, ((l2 + 1)*l3), 1, 0.0).reshape(l2+1, l3);
		xxx += (l2 + 1)*l3;
		Matrix w3 = vv.subMatrix(xxx, 0, ((l3 + 1)*l4), 1, 0.0).reshape(l3+1, l4);
		xxx += (l3 + 1)*l4;
		Matrix wClass = vv.subMatrix(xxx, 0, ((l4 + 1)*l5), 1, 0.0).reshape(l4+1, l5);
		
		xx = xx.spliceInRowType(Matrix.ones(n, 1));

		Matrix w1probs = Matrix.ones(xx.m, w1.n).dotDiv(xx.multi(w1).multi(-1.0).exp().add(1.0)).spliceInRowType(Matrix.ones(n, 1));
		Matrix w2probs = Matrix.ones(w1probs.m, w2.n).dotDiv(w1probs.multi(w2).multi(-1.0).exp().add(1.0)).spliceInRowType(Matrix.ones(n, 1));
		Matrix w3probs = Matrix.ones(w2probs.m, w3.n).dotDiv(w2probs.multi(w3).multi(-1.0).exp().add(1.0)).spliceInRowType(Matrix.ones(n, 1));
		
		Matrix targetOut = w3probs.multi(wClass).exp();
		targetOut = targetOut.dotDiv(targetOut.rowSum().repmat(1, 10));
		Matrix targetExpand = Matrix.zeros(targetOut.m, targetOut.n);
		for(int i = 0; i < target.m; ++i) {
			targetExpand.putValue(1.0, i, (int)target.data[i]);
		}
		double f = -1.0 * targetExpand.dotMulti(targetOut.log()).sum();
		
		Matrix io = targetOut.dotMinus(targetExpand);
		Matrix dwClass = w3probs.trans().multi(io);
		
		Matrix ix3 = io.multi(wClass.trans()).dotMulti(w3probs).dotMulti(Matrix.ones(w3probs.m, w3probs.n).dotMinus(w3probs));
		ix3 = ix3.subMatrix(0, 0, ix3.m, (ix3.n - 1), 0.0);
		Matrix dw3 = w2probs.trans().multi(ix3);
		
		Matrix ix2 = ix3.multi(w3.trans()).dotMulti(w2probs).dotMulti(Matrix.ones(w2probs.m, w2probs.n).dotMinus(w2probs));
		ix2 = ix2.subMatrix(0, 0, ix2.m, (ix2.n - 1), 0.0);
		Matrix dw2 = w1probs.trans().multi(ix2);
		
		Matrix ix1 = ix2.multi(w2.trans()).dotMulti(w1probs).dotMulti(Matrix.ones(w1probs.m, w1probs.n).dotMinus(w1probs));
		ix1 = ix1.subMatrix(0, 0, ix1.m, (ix1.n - 1), 0.0);
		Matrix dw1 = xx.trans().multi(ix1);
		
		Matrix df = dw1.reshape(dw1.m*dw1.n, 1).spliceInRankType(dw2.reshape(dw2.m*dw2.n, 1)).spliceInRankType(dw3.reshape(dw3.m*dw3.n, 1)).spliceInRankType(dwClass.reshape(dwClass.m*dwClass.n, 1));
		
		return new CGResult(f, df);
	}
	
	private static CGResult cgClassfiyInit(Matrix vv, Matrix dim, Matrix w3probs, Matrix target) {
		int l1 = (int) dim.data[0];
		int l2 = (int) dim.data[1];
		
		int n = w3probs.m;
		
		Matrix wClass = vv.reshape((l1 + 1), l2);
		w3probs = w3probs.spliceInRowType(Matrix.ones(n, 1));
		
		Matrix targetOut = w3probs.multi(wClass).exp();
		targetOut = targetOut.dotDiv(targetOut.rowSum().repmat(1, 10));
		
		Matrix targetExpand = Matrix.zeros(targetOut.m, targetOut.n);
		for(int i = 0; i < target.m; ++i) {
			targetExpand.putValue(1.0, i, (int)target.data[i]);
		}
		
		double f  = -1.0 * targetExpand.dotMulti(targetOut.log()).sum();
		Matrix io = targetOut.dotMinus(targetExpand);
		
		Matrix dwClass = w3probs.trans().multi(io);
		Matrix df = dwClass.reshape(dwClass.m*dwClass.n, 1);
		
		return new CGResult(f, df);
	}
	
	private static CGResult cgMnist(Matrix vv, Matrix dim, Matrix xx) {
		int l[] = new int[9];
		for(int i = 0; i < l.length; ++i) {
			l[i] = (int) dim.data[i];
		}
		int n = xx.m;
		
		Matrix w[] = new Matrix[8];
		int xxx = 0;
		for(int i = 0; i < 8; ++i) {
			w[i] = vv.subMatrix(xxx, 0, (l[i] + 1)*l[i+1], 1, 0.0).reshape(l[i]+1, l[i+1]);
			xxx += (l[i] + 1)*l[i+1];
		}
		
		xx = xx.spliceInRowType(Matrix.ones(n, 1));
		
		Matrix w1probs = Matrix.ones(xx.m, w[0].n).dotDiv(xx.multi(w[0]).multi(-1.0).exp().add(1.0)).spliceInRowType(Matrix.ones(n, 1));
		Matrix w2probs = Matrix.ones(w1probs.m, w[1].n).dotDiv(w1probs.multi(w[1]).multi(-1.0).exp().add(1.0)).spliceInRowType(Matrix.ones(n, 1));
		Matrix w3probs = Matrix.ones(w2probs.m, w[2].n).dotDiv(w2probs.multi(w[2]).multi(-1.0).exp().add(1.0)).spliceInRowType(Matrix.ones(n, 1));
		Matrix w4probs = w3probs.multi(w[3]).spliceInRowType(Matrix.ones(n, 1));
		Matrix w5probs = Matrix.ones(w4probs.m, w[4].n).dotDiv(w4probs.multi(w[4]).multi(-1.0).exp().add(1.0)).spliceInRowType(Matrix.ones(n, 1));
		Matrix w6probs = Matrix.ones(w5probs.m, w[5].n).dotDiv(w5probs.multi(w[5]).multi(-1.0).exp().add(1.0)).spliceInRowType(Matrix.ones(n, 1));
		Matrix w7probs = Matrix.ones(w6probs.m, w[6].n).dotDiv(w6probs.multi(w[6]).multi(-1.0).exp().add(1.0)).spliceInRowType(Matrix.ones(n, 1));
		Matrix xxOut   = Matrix.ones(w7probs.m, w[7].n).dotDiv(w7probs.multi(w[7]).multi(-1.0).exp().add(1.0));
		
		double f   = (-1.0 / n) * xx.subMatrix(0, 0, xx.m, (xx.n - 1), 0.0).dotMulti(xxOut.log()).dotAdd(xx.subMatrix(0, 0, xx.m, (xx.n - 1), 0.0).multi(-1.0).add(1.0).dotMulti(Matrix.ones(xxOut.m, xxOut.n).dotMinus(xxOut).log())).sum();
		Matrix io  = xxOut.dotMinus(xx.subMatrix(0, 0, xx.m, (xx.n - 1), 0.0)).multi((1.0 / n));
		Matrix ix8 = new Matrix(io); 
		Matrix dw8 = w7probs.trans().multi(io);
		
		Matrix ix7 = ix8.multi(w[7].trans()).dotMulti(w7probs).dotMulti(Matrix.ones(w7probs.m, w7probs.n).dotMinus(w7probs));
		ix7 = ix7.subMatrix(0, 0, ix7.m, (ix7.n - 1), 0.0);
		Matrix dw7 = w6probs.trans().multi(ix7);
		
		Matrix ix6 = ix7.multi(w[6].trans()).dotMulti(w6probs).dotMulti(Matrix.ones(w6probs.m, w6probs.n).dotMinus(w6probs));
		ix6 = ix6.subMatrix(0, 0, ix6.m, (ix6.n - 1), 0.0);
		Matrix dw6 = w5probs.trans().multi(ix6);
		
		Matrix ix5 = ix6.multi(w[5].trans()).dotMulti(w5probs).dotMulti(Matrix.ones(w5probs.m, w5probs.n).dotMinus(w5probs));
		ix5 = ix5.subMatrix(0, 0, ix5.m, (ix5.n - 1), 0.0);
		Matrix dw5 = w4probs.trans().multi(ix5);

		Matrix ix4 = ix5.multi(w[4].trans());
		ix4 = ix4.subMatrix(0, 0, ix4.m, (ix4.n - 1), 0.0);
		Matrix dw4 = w3probs.trans().multi(ix4);

		Matrix ix3 = ix4.multi(w[3].trans()).dotMulti(w3probs).dotMulti(Matrix.ones(w3probs.m, w3probs.n).dotMinus(w3probs));
		ix3 = ix3.subMatrix(0, 0, ix3.m, (ix3.n - 1), 0.0);
		Matrix dw3 = w2probs.trans().multi(ix3);

		Matrix ix2 = ix3.multi(w[2].trans()).dotMulti(w2probs).dotMulti(Matrix.ones(w2probs.m, w2probs.n).dotMinus(w2probs));
		ix2 = ix2.subMatrix(0, 0, ix2.m, (ix2.n - 1), 0.0);
		Matrix dw2 = w1probs.trans().multi(ix2);

		Matrix ix1 = ix2.multi(w[1].trans()).dotMulti(w1probs).dotMulti(Matrix.ones(w1probs.m, w1probs.n).dotMinus(w1probs));
		ix1 = ix1.subMatrix(0, 0, ix1.m, (ix1.n - 1), 0.0);
		Matrix dw1 = xx.trans().multi(ix1);

		Matrix df = dw1.reshape(dw1.m*dw1.n, 1).spliceInRankType(dw2.reshape(dw2.m*dw2.n, 1)).spliceInRankType(dw3.reshape(dw3.m*dw3.n, 1)).spliceInRankType(dw4.reshape(dw4.m*dw4.n, 1));
		df = df.spliceInRankType(dw5.reshape(dw5.m*dw5.n, 1)).spliceInRankType(dw6.reshape(dw6.m*dw6.n, 1)).spliceInRankType(dw7.reshape(dw7.m*dw7.n, 1)).spliceInRankType(dw8.reshape(dw8.m*dw8.n, 1));
		
		return new CGResult(f, df);
	}
	
	public static CGResult exec(CGFunc cgf, EnumMap<CGFunc, Map<String, Matrix>> args) throws Exception {
		if(!args.isEmpty()) {
			CGResult res = null;
			
			switch(cgf) {
			case CG_CLASSIFY: {
				if(args.containsKey(cgf)) {
					Map<String, Matrix> params = args.get(cgf);

					Matrix vv  = params.get("vv");
					Matrix dim = params.get("dim");
					Matrix xx  = params.get("xx");
					Matrix target = params.get("target");
					
					res = ConjugateGradient.cgClassfiy(vv, dim, xx, target);
				} else {
					throw new Exception("There are no parameters for function CG_CLASSIFY.");
				}
				
				break;
			}
			case CG_CLASSIFY_INIT: {
				if(args.containsKey(cgf)) {
					Map<String, Matrix> params = args.get(cgf);

					Matrix vv  = params.get("vv");
					Matrix dim = params.get("dim");
					Matrix w3probs  = params.get("w3probs");
					Matrix target = params.get("target");
					
					res = ConjugateGradient.cgClassfiyInit(vv, dim, w3probs, target);				
				} else {
					throw new Exception("There are no parameters for function CG_CLASSIFY_INIT.");
				}
				
				break;
			}
			case CG_MNIST: {
				if(args.containsKey(cgf)) {
					Map<String, Matrix> params = args.get(cgf);

					Matrix vv  = params.get("vv");
					Matrix dim = params.get("dim");
					Matrix xx  = params.get("xx");
					
					res = ConjugateGradient.cgMnist(vv, dim, xx);				
				} else {
					throw new Exception("There are no parameters for function CG_MNIST.");
				}
				
				break;				
			}
			default: {
				throw new Exception("Could not find Realize Function for Conjugate Gradient Optimizing.");
			}
			}
			
			return res;
		}
		
		return null;
	}
	
}
