package com.utils;

public class ByteToInteger {
	
	public static int[] convert(byte[] dat, int size) {
		if(size >= 1 && size <= 4) {
			if(dat.length % size == 0) {
				int n = dat.length / size;

				int res[] = new int[n];
				for(int i = 0; i < n; ++i) {
					res[i] = 0x00000000;
				}
				
				for(int i = 0; i < n; ++i) {
					for(int j = 0; j < size; ++j) {
						int digit = 0x000000ff & dat[i*size + j];
						res[i] |= digit;
						res[i] <<= (size - j - 1)*8;
					}
				}
				
				return res;
			}
		}
		
		return null;
	}

	public static void main(String[] args) {
	}

}
