package com.utils;

public class ByteArrayToDataArray {

	public static Object[] convert(byte[] data, Object type) {
		if(data != null) {
			if(type instanceof Short) {
				int n = data.length / 2;
				Short[] res = new Short[n];
				
				for(int i = 0; i < n; ++i) {
					short tmp =  0x0000;
					
					tmp = (short) ((tmp | (data[i*2]     & 0x00ff)) << 8);
					tmp = (short) ((tmp | (data[i*2 + 1] & 0x00ff)));
					
					res[i] = tmp;
				}
				
				return res;
			} else if(type instanceof Character) {
				int n = data.length / 2;
				Character[] res = new Character[n];
				
				for(int i = 0; i < n; ++i) {
					char tmp =  0x0000;

					tmp = (char) ((tmp | (data[i*2]     & 0x00ff)) << 8);
					tmp = (char) ((tmp | (data[i*2 + 1] & 0x00ff)));
					
					res[i] = tmp;
				}
				
				return res;
			} else if(type instanceof Integer) {
				int n = data.length / 4;
				Integer[] res = new Integer[n];

				for(int i = 0; i < n; ++i) {
					int tmp =  0x00000000;
					
					tmp = (int) (tmp | ((data[i*4]     & 0x000000ff) << 24));
					tmp = (int) (tmp | ((data[i*4 + 1] & 0x000000ff) << 16));
					tmp = (int) (tmp | ((data[i*4 + 2] & 0x000000ff) << 8));
					tmp = (int) (tmp | ((data[i*4 + 3] & 0x000000ff)));
					
					res[i] = tmp;
				}
				
				return res;
			} else if(type instanceof Float) {
				int n = data.length / 4;
				Float[] res = new Float[n];

				for(int i = 0; i < n; ++i) {
					int tmp =  0x00000000;

					tmp = (int) (tmp | ((data[i*4]    & 0x000000ff)  << 24));
					tmp = (int) (tmp | ((data[i*4 + 1] & 0x000000ff) << 16));
					tmp = (int) (tmp | ((data[i*4 + 2] & 0x000000ff) << 8));
					tmp = (int) (tmp | ((data[i*4 + 3] & 0x000000ff)));
					
					res[i] = Float.intBitsToFloat(tmp);
				}
				
				return res;
			} else if(type instanceof Long) {
				int n = data.length / 8;
				Long[] res = new Long[n];
				
				for(int i = 0; i < n; ++i) {
					long tmp = 0x0000000000000000L;
					
					tmp = (long) (((data[i*8]     & 0x00000000000000ffL) << 56) | tmp);
					tmp = (long) (((data[i*8 + 1] & 0x00000000000000ffL) << 48) | tmp);
					tmp = (long) (((data[i*8 + 2] & 0x00000000000000ffL) << 40) | tmp);
					tmp = (long) (((data[i*8 + 3] & 0x00000000000000ffL) << 32) | tmp);
					tmp = (long) (((data[i*8 + 4] & 0x00000000000000ffL) << 24) | tmp);
					tmp = (long) (((data[i*8 + 5] & 0x00000000000000ffL) << 16) | tmp);
					tmp = (long) (((data[i*8 + 6] & 0x00000000000000ffL) << 8)  | tmp);
					tmp = (long) ((data[i*8 + 7]  & 0x00000000000000ffL)        | tmp);

					res[i] = tmp;
				}
				
				return res;
			} else if(type instanceof Double) {
				int n = data.length / 8;
				Double[] res = new Double[n];
				
				for(int i = 0; i < n; ++i) {
					long tmp = 0x0000000000000000L;
					
					tmp = (long) (((data[i*8]     & 0x00000000000000ffL) << 56) | tmp);
					tmp = (long) (((data[i*8 + 1] & 0x00000000000000ffL) << 48) | tmp);
					tmp = (long) (((data[i*8 + 2] & 0x00000000000000ffL) << 40) | tmp);
					tmp = (long) (((data[i*8 + 3] & 0x00000000000000ffL) << 32) | tmp);
					tmp = (long) (((data[i*8 + 4] & 0x00000000000000ffL) << 24) | tmp);
					tmp = (long) (((data[i*8 + 5] & 0x00000000000000ffL) << 16) | tmp);
					tmp = (long) (((data[i*8 + 6] & 0x00000000000000ffL) << 8)  | tmp);
					tmp = (long) ((data[i*8 + 7]  & 0x00000000000000ffL)        | tmp);

					res[i] = Double.longBitsToDouble(tmp);
				}
				
				return res;
			} else {
				return null;
			}			
		} else {
			return null;
		}
	}
		
	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

}
