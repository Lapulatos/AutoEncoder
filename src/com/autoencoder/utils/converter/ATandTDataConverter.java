package com.autoencoder.utils.converter;

import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.List;

import javax.imageio.ImageIO;

import com.utils.math.Matrix;

import eugfc.imageio.plugins.PNMRegistry;

public class ATandTDataConverter extends DataConverter {

	static {
		PNMRegistry.registerAllServicesProviders();
	}
	
	public static final int CLASS_NUM = 40;
	public static final int NUM_PER_CLASS = 10;
	
	private String   datFolder;	
	private String[] subFolder;
	
	private int totalClass;
	
	public ATandTDataConverter() {
		this.totalRead = 0;
		
		this.datWidth = 0;
		this.datHeight = 0;
		
		this.datNumber = 0;
		this.lblNumber = 0;
		
		this.datFolder = "";
		this.subFolder = null;
		
		this.totalClass = 0;
		this.data = null;
	}
	
	public ATandTDataConverter(String datFolder) {
		this.datFolder = datFolder;
		this.subFolder = new String[ATandTDataConverter.CLASS_NUM];
	}
	
	@Override
	public void init() {
		try {
			File df = new File(this.datFolder);
			
			if(df.isDirectory()) {
				File[] flist = df.listFiles();
				
				int count = 0;
				for(int i = 0; i < flist.length; ++i) {
					if(flist[i].isDirectory()) {
						this.subFolder[count++] = flist[i].getPath();
					}
				}
				this.totalClass = count;
				
				if(count == 0) {
					throw new Exception("There are not subfolder in AT & T DataSet.");
				}
				
				this.data = new ArrayList<ArrayList<Matrix>>();
				for(int i = 0; i < this.totalClass; ++i) {
					this.data.add(new ArrayList<Matrix>());
				}
				
//				for(int i = 0; i < ATandTDataConverter.CLASS_NUM; ++i) {
//					System.out.println(this.subFolder[i]);
//				}				
//				System.out.println("total class:" + this.data.size());
			}
		} catch(Exception e) {
			e.printStackTrace();
		}
	}

	@Override
	public void read() {
		try {
			for(int i = 0; i < this.subFolder.length; ++i) {
				int idx = Integer.parseInt(this.subFolder[i].substring(this.subFolder[i].lastIndexOf("s") + 1)) - 1;
				
				File dfi = new File(this.subFolder[i]);
				if(dfi.isDirectory()) {
					File[] flist = dfi.listFiles();
					
					for(int j = 0; j < flist.length; ++j) {
						if(flist[j].getName().endsWith(".pgm")) {
							BufferedImage bi = ImageIO.read(new File(flist[j].getPath()));
							
							int height = bi.getHeight();
							int width  = bi.getWidth();
							int[] img = bi.getRGB(0, 0, width, height, null, 0, width);
							
							this.data.get(idx).add(new Matrix(img, height, width).nearestInterpolation(23, 28));
						}
					}
				}
			}
			
			
			int count = 0;
			for(int i = 0; i < this.data.size(); ++i) {
				count += this.data.get(i).size();
			}
			this.datNumber = count;
			this.totalRead = count;
			this.lblNumber = count;
			
			
			this.datHeight = this.data.get(0).get(0).m;
			this.datWidth  = this.data.get(0).get(0).n;

		} catch(Exception e) {
			e.printStackTrace();
		}
	}	
	
	protected int getTotalClass() {
		return totalClass;
	}

	public static void main(String[] args) {
		ATandTDataConverter attdc = new ATandTDataConverter("C:/Users/sujie/Desktop/GDPCA/dataset/AT&T/");
		attdc.init();
		attdc.read();
	}

}
