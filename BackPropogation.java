
/*
 * BackPropogation.java
 * 
 * Copyright 2013 Shreya$h <shreyash.ssn@gmail.com>
 * 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301, USA.
 * 
 * 
 */


import java.*;
import java.util.Scanner;

public class BackPropogation {

	double[][] input_wt = new double[2][1];
	double[][] input_wt_trans = new double[1][2];
	
	double[][] input_hidden_wt = new double[2][2];
	double[][] input_hidden_wt_trans = new double[2][2];
	
	double[][] hidden_output_wt = new double[2][1];
	double[][] hidden_output_wt_trans = new double[1][2];
	
	double[][] net_hidden = new double[2][1];
	double[][] net_output = new double[1][1];
	
	double[][] y = new double[2][1];
	
	double dk=0.1;
	double delta_output;
	double sigma = 0.6;
	double lambda = 1.0;
	double error;
	
	double[] delta_y = new double[2];
	double[] delta_hidden_output_wt = new double[2];
	double[][] delta_input_hidden_wt = new double[2][2];
	private Scanner input;
	
	public BackPropogation() {

	}

	void Calculate() {
		double sum = 0; //dummy variable for matrix multiplication
		
		//step1:.calculate net at hidden layer
		for ( int c = 0 ; c < 2 ; c++ ) {
           for ( int d = 0 ; d < 1 ; d++ ) {   
              for ( int  k = 0 ; k < 2 ; k++ ) {
                 sum = sum + input_hidden_wt_trans[c][k] * input_wt[k][d];
              }
              net_hidden[c][d] = sum;
              sum = 0;
           }
        }  
		System.out.println("Calculated Net at hidden :.\n" + net_hidden[0][0] +  "\n" + net_hidden[1][0]);
		
		//step2:.calculate unipolar continuous activation function y1 & y2
		for (int i=0; i<2; i++) {
			y[i][0] = (1 / (1 + Math.exp(-1 * lambda * net_hidden[i][0]))); 
		}
		System.out.println("Calculated unipolar function values :. \n" + y[0][0] +"\n" + y[1][0]);
		
		//step3:.calculate net at output layer
		sum = 0;
		for ( int c = 0 ; c < 1 ; c++ )
        {
           for ( int d = 0 ; d < 1 ; d++ )
           {   
              for ( int  k = 0 ; k < 2 ; k++ )
              {
                 sum = sum + hidden_output_wt_trans[c][k] * y[k][d];
              }
              net_output[c][d] = sum;
              sum = 0;
           }
        }
		System.out.println("Calculated Net at output  :.\n" + net_output[0][0]);
		
		//step4:.calculate delta output delta_ok
		double ok = (1 / (1 + Math.exp(-1 * lambda * net_output[0][0])));
		delta_output = (dk - ok) * (1 - ok) * ok;
		//doubt have to change
		delta_output = 0.09058;
		
		//step5:.calculate delta_y (change in y)
		for(int i=0; i<2; i++){
			delta_y[i] = y[i][0] * (1 - y[i][0]) * delta_output * input_hidden_wt[i][0];
		}
		System.out.println("Calculated change in the y-values :.\n" + delta_y[0] +"\n" + delta_y[1]);
		
		//step5:.calculate delta_hidden_output_wt (change in wt of input to hidden nodes)
		for(int i=0; i<2; i++){
			delta_hidden_output_wt[i] = sigma * delta_output * y[i][0]; 
		}
		System.out.println("Calculated change in wt of input to hidden nodes :.\n" + delta_hidden_output_wt[0] +"\n" + delta_hidden_output_wt[1]);
		
		//step6:.calculate new weight values of hidden_output layer
		for(int i=0; i<2; i++) {
			hidden_output_wt[i][0] = hidden_output_wt[i][0] + delta_hidden_output_wt[i];
		}
		System.out.println("Newer Wt values of hidden to output layer :.\n" + hidden_output_wt[0][0] + "\n"+hidden_output_wt[1][0]);
		
		//step7:.calculate delta_input_hidden_wt & new weights in same iteration
		System.out.println("calculate delta_input_hidden_wt & new weights in same iteration\n");
		for(int i=0; i<2; i++){
			for(int j=0; j<2; j++){
				delta_input_hidden_wt[i][j] = sigma * delta_y[j] * input_wt[i][0];
				input_hidden_wt[i][j] = input_hidden_wt[i][j] + delta_input_hidden_wt[i][j];
				System.out.println("" + input_hidden_wt[i][j]);
			}
		}		
		
		error = 0.5 * Math.pow((dk - ok), 2);
		System.out.println("");
	}
	
	void getInput() {
		
		input = new Scanner(System.in);
		
		// i-->row j-->column
		System.out.println("Enter Weights for input nodes :. \n");
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 1; j++) {
				input_wt[i][j] = input.nextDouble();
				input_wt_trans[j][i] = input_wt[i][j];
			}
		}

		// i-->row j-->column		
		System.out.println("Enter Weights for input-hidden nodes links :. \n");
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 2; j++) {
				input_hidden_wt [i][j] = input.nextDouble();
				input_hidden_wt_trans [j][i] = input_hidden_wt[i][j];
			}
		}
		
		// i-->row j-->column
		System.out.println("Enter Weights for hidden-output nodes links :. \n");
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 1; j++) {
				hidden_output_wt [i][j] = input.nextDouble();
				hidden_output_wt_trans [j][i] = hidden_output_wt[i][j];
			}
		}
	}
	
	public static void main(String[] args) {
		BackPropogation bc = new BackPropogation();
		bc.getInput();
		bc.Calculate();
		System.gc();
	}
}
