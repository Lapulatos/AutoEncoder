package com.utils.img;

import java.awt.EventQueue;
import java.awt.Graphics;

import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.border.EmptyBorder;

import com.utils.math.Matrix;
import javax.swing.JLabel;
import java.awt.Font;

public class ProcessResultImageFrame extends JFrame {

	private static final long serialVersionUID = 1L;
	private JPanel contentPane;
	
	public ImagePanel real[] = null;
	public ImagePanel rec[]  = null;

	public static final int numPic = 15;
	
	private Matrix realData;
	private Matrix recData;
	
	private final int imgPanelHeight = 80;
	private final int imgPanelWidth  = 80;
	
	private int realH;
	private int realW;
	private int recH;
	private int recW;
	
	/**
	 * Launch the application.
	 */
	public static void main(String[] args) {
		EventQueue.invokeLater(new Runnable() {
			public void run() {
				try {
					Matrix realdata = Matrix.rand(15, 10000);
					Matrix recdata  = Matrix.rand(15, 10000);

					ProcessResultImageFrame frame = new ProcessResultImageFrame();
					
					frame.setRealH(100);
					frame.setRealW(100);
					frame.setRecH(100);
					frame.setRecW(100);
					
					frame.setRealData(realdata);
					frame.setRecData(recdata);
					
					frame.setVisible(true);
					frame.setResizable(false);
					frame.paint();
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		});
	}

	/**
	 * Create the frame.
	 */
	
	public ProcessResultImageFrame() {		
		setTitle("Process Result");
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		setBounds(100, 100, 1363, 292);
		contentPane = new JPanel();
		contentPane.setBorder(new EmptyBorder(5, 5, 5, 5));
		setContentPane(contentPane);
		contentPane.setLayout(null);
		
		this.real = new ImagePanel[numPic];
		this.rec  = new ImagePanel[numPic];
		
		int styReal = 40, styRec = styReal + imgPanelHeight + 10;
		int stxReal = 10, stxRec = stxReal;
		
		for(int i = 0; i < numPic; ++i) {
			this.real[i] = new ImagePanel();
			this.real[i].setBounds(stxReal + i*(imgPanelWidth + 10), styReal, imgPanelWidth, imgPanelHeight);
			this.contentPane.add(this.real[i]);
			
			this.rec[i] = new ImagePanel();
			this.rec[i].setBounds(stxRec + i*(imgPanelWidth + 10), styRec, imgPanelWidth, imgPanelHeight);
			this.contentPane.add(this.rec[i]);
		}
		
		JLabel lblReal = new JLabel("Real Data");
		lblReal.setFont(new Font("ËÎÌו", Font.PLAIN, 16));
		lblReal.setBounds(642, 10, 80, 24);
		contentPane.add(lblReal);
		
		JLabel lblRec = new JLabel("Reconstructed Data");
		lblRec.setFont(new Font("ËÎÌו", Font.PLAIN, 16));
		lblRec.setBounds(612, 217, 159, 15);
		contentPane.add(lblRec);
		
	}
	
	public void setRealH(int realH) {
		this.realH = realH;
	}

	public void setRealW(int realW) {
		this.realW = realW;
	}

	public void setRecH(int recH) {
		this.recH = recH;
	}

	public void setRecW(int recW) {
		this.recW = recW;
	}

	public void setRealData(Matrix realData) {
		this.realData = realData;
	}

	public void setRecData(Matrix recData) {
		this.recData = recData;
	}

	public void paint() {
		if((this.realData != null && this.realData.m != 0 && this.realData.n != 0 && this.realData.m >= numPic) && (this.recData != null && this.recData.m != 0 && this.recData.n != 0 && this.recData.m >= numPic)) {
			int realDataN = this.realData.n;
			int recDataN = this.recData.n;
			
			for(int i = 0; i < numPic; ++i) {
				this.real[i].setDataMatrix(this.realData.subMatrix(i, 0, 1, realDataN, 0.0).reshape(this.realH, this.realW).nearestInterpolation(imgPanelWidth, imgPanelHeight));
				this.real[i].paint(this.real[i].getGraphics());

				this.rec[i].setDataMatrix(this.recData.subMatrix(i, 0, 1, recDataN, 0.0).reshape(this.recH, this.recW).nearestInterpolation(imgPanelWidth, imgPanelHeight));
				this.rec[i].paint(this.rec[i].getGraphics());
			}
		}
	}
}
