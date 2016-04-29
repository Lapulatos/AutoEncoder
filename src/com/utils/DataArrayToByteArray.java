package com.utils;

public class DataArrayToByteArray {
	
	public static byte[] convert(short[] data) {
		if(data != null) {
			int n = data.length;
			byte[] res = new byte[n*2];
			
			for(int i = 0; i < n; ++i) {
				res[i*2]     = (byte) ((data[i] & 0xff00) >> 8);
				res[i*2 + 1] = (byte) ((data[i] & 0x00ff) );
			}
			
			return res;
		} else {
			return null;
		}
	}
	
	public static byte[] convert(char[] data) {
		if(data != null) {
			int n = data.length;
			byte[] res = new byte[n*2];
			
			for(int i = 0; i < n; ++i) {
				res[i*2]     = (byte) ((data[i] & 0xff00) >> 8);
				res[i*2 + 1] = (byte) ((data[i] & 0x00ff) );
			}
			
			return res;
		} else {
			return null;
		}
	}
	
	public static byte[] convert(int[] data) {
		if(data != null) {
			int n = data.length;
			byte[] res = new byte[n*4];
			
			for(int i = 0; i < n; ++i) {
				res[i*4]     = (byte) ((data[i] & 0xff000000) >> 24);
				res[i*4 + 1] = (byte) ((data[i] & 0x00ff0000) >> 16);
				res[i*4 + 2] = (byte) ((data[i] & 0x0000ff00) >> 8);
				res[i*4 + 3] = (byte) ((data[i] & 0x000000ff) );
			}
			
			return res;
		} else {
			return null;
		}
	}
		
	public static byte[] convert(float[] data) {
		if(data != null) {
			int n = data.length;
			int[] tmp = new int[data.length];
			
			for(int i = 0; i < n; ++i) {
				tmp[i] = Float.floatToIntBits(data[i]);
			}
			return convert(tmp);
		} else {
			return null;
		}
	}
		
	public static byte[] convert(long[] data) {
		if(data != null) {
			int n = data.length;
			byte[] res = new byte[n*8];
			
			for(int i = 0; i < n; ++i) {
				res[i*8]         = (byte) ((data[i] >>> 56) & 0xff);
				res[i*8 + 1]     = (byte) ((data[i] >>> 48) & 0xff);
				res[i*8 + 2]     = (byte) ((data[i] >>> 40) & 0xff);
				res[i*8 + 3]     = (byte) ((data[i] >>> 32) & 0xff);
				res[i*8 + 4]     = (byte) ((data[i] >>> 24) & 0xff);
				res[i*8 + 5]     = (byte) ((data[i] >>> 16) & 0xff);
				res[i*8 + 6]     = (byte) ((data[i] >>> 8) & 0xff);
				res[i*8 + 7]     = (byte) ((data[i] ) & 0xff);
			}
			
			return res;
		} else {
			return null;
		}
	}
	
	public static byte[] convert(double[] data) {
		if(data != null) {
			int n = data.length;
			long[] tmp = new long[data.length];
			
			for(int i = 0; i < n; ++i) {
				tmp[i] = Double.doubleToLongBits(data[i]);
			}
			return convert(tmp);
		} else {
			return null;
		}
	}

	public static void main(String[] args) {
		char[] a = new char[3];
		a[0] = (char) 'A';
		a[1] = (char) '-';
		a[2] = (char) '/';
		
		byte[] res = DataArrayToByteArray.convert(a);
		Object[] cres = ByteArrayToDataArray.convert(res, new Character((char) 0));
		Character[]  cfres = (Character[])cres;
		
		for(int i = 0; i < cfres.length; ++i) {
			System.out.print(cfres[i] + " ");
		}
		
		System.out.println();
		
	}

}
