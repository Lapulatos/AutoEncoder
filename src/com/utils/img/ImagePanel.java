package com.utils.img;

import java.awt.Graphics;
import java.awt.image.BufferedImage;

import javax.swing.JPanel;

import com.utils.math.Matrix;

public class ImagePanel extends JPanel {

	private static final long serialVersionUID = 1L;

	private Matrix dataMatrix;
	
	public ImagePanel() {
		this.dataMatrix = null;
	}
	
	public ImagePanel(Matrix data) {
		this.dataMatrix = data;
	}
	
	
	public Matrix getDataMatrix() {
		return dataMatrix;
	}
	



	public void setDataMatrix(Matrix dataMatrix) {
		this.dataMatrix = dataMatrix;
	}
	



	@Override
	public void paint(Graphics g) {
		if((this.dataMatrix != null) && (this.dataMatrix.m > 0) && (this.dataMatrix.n > 0)) {
			BufferedImage bf = new BufferedImage(dataMatrix.m, dataMatrix.n, BufferedImage.TYPE_INT_RGB);

			bf.setRGB(0, 0, this.dataMatrix.n, this.dataMatrix.m, extendChannel(this.dataMatrix, 255.0), 0, this.dataMatrix.n);
			g.drawImage(bf, 0, 0, null, this);
		}
	}
	
	public int[] extendChannel(Matrix src, double scale) {
		src = src.multi(scale);
		int[] res = src.getDataTypeInt();
		
		for(int i = 0; i < res.length; ++i) {
			res[i] = res[i] & 0x000000ff;
			int digit = res[i];
			
			res[i] = ((res[i] << 8) | digit);
			res[i] = ((res[i] << 8) | digit);
		}
		
		return res;
	}
	
}
