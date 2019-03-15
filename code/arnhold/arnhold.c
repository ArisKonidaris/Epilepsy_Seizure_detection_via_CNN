/*-----------------------------------------------------------------------------------
*  INPUTS:	
*          X : timeSteps X channels data matrix (passed from Matlab)				
*          Time delay                   TAU										
*          Embedding dimension          EMBED			
*          Number of nearest neighbours NN			
*          Theiler correction           THEILER											
*  SETUP:  In Matlab run "mex arnhold.c", selecting [1]Lcc or [2]Microsoft Visual C/C++ 
*          To execute, type "arnhold(X, TAU, EMBED, NN, THEILER);"				
*  OUTPUT: S : channels X channels interdependence	
/--------------------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include "mex.h"
#include "matrix.h"

void calculateInterdependencies(double **S, double **X, int channels, int timeSteps, int TAU, int EMBED, int NN, int THEILER);

double standardDeviation(double *x, int length);

double *new_darray(int cols);
void free_darray(double *array);
double **new_dmatrix(int rows, int cols);
void free_dmatrix(double **matrix, int rows);

int *new_iarray(int cols);
void free_iarray(int *array);
int **new_imatrix(int rows, int cols);
void free_imatrix(int **matrix, int rows);


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{   
   double **X, **S, *in_ptr, *out_ptr;
   int TAU, EMBED, NN, THEILER;
   int timeSteps, channels;
   int i, j, ic;

   if(nrhs != 5) 
   { 
      mexErrMsgTxt("Usage: arnhold(X, TAU, EMBED, NN, THEILER)");
   }
	
   in_ptr = mxGetPr(prhs[0]);			
   timeSteps = mxGetM(prhs[0]); 		
   channels = mxGetN(prhs[0]);	

   TAU		= mxGetScalar(prhs[1]);	
   EMBED	= mxGetScalar(prhs[2]);
   NN		= mxGetScalar(prhs[3]);
   THEILER	= mxGetScalar(prhs[4]);

   plhs[0] = mxCreateDoubleMatrix(channels, channels, mxREAL);
   out_ptr = mxGetPr(plhs[0]);	

   X = new_dmatrix(channels, timeSteps);
   S = new_dmatrix(channels, channels);

   for(i = 0; i < channels; i++) {
      ic = i*timeSteps;  
      for(j = 0; j < timeSteps; j++)
         X[i][j] = in_ptr[ic + j];
   }

   calculateInterdependencies(S, X, channels, timeSteps, TAU, EMBED, NN, THEILER);	

   for(i = 0; i < channels; i++) {
      ic = i*channels;  
      for(j = 0; j < channels; j++)
         out_ptr[ic + j] = S[i][j];
   }


   free_dmatrix(S, channels); 
   free_dmatrix(X, channels);

   return;
}

/*----------------- End of mexFunction gateway routine ------------------*/

/* --------- function NON-LINEAR INTERDEPENDENCIES ----------- */

void calculateInterdependencies(double **S, double **X, int channels, int NDAT, int TAU, int EMBED, int NN, int THEILER)
{
    double /*rxx, ryy,*/ rxy, ryx, rrx, rry, *rr;
    double **aux, *auxx, *auxy, **dist, *distx, *disty;
    double **nxy, *x;
    int i, j, k, ch, ch1, ch2, **index, *indexx, *indexy;
    int numDelayVectors = NDAT-(EMBED-1)*TAU;
    double xij, distxij;
    int numPoints; 
    int EMBED_X_TAU = EMBED*TAU; 
	
    aux = new_dmatrix(channels, NN+1);
    index = new_imatrix(channels, NN+1); 
    dist = new_dmatrix(channels, NDAT);
    rr = new_darray(channels);
    nxy = new_dmatrix(channels, channels);    

    for(ch1 = 0; ch1 < channels; ch1++)
        for(ch2 = ch1+1; ch2 < channels; ch2++)
            nxy[ch1][ch2] = 0;

    for(i = 0; i < numDelayVectors; i++) 
    {
        for(ch = 0; ch < channels; ch++) {
            auxx   = aux[ch];
            indexx = index[ch];
            distx  = dist[ch]; 
            x = X[ch];
            for(k = 0; k < NN; k++)   
	    {
                auxx[k] = DBL_MAX;
	        indexx[k] = 10000000;
	    }
		
	    auxx[NN] = 0.0;
	    indexx[NN] = 10000000;

	    rrx = 0;
            numPoints = 0;  
	    for(j = 0; j < numDelayVectors; j++) 
	    {            
                if ((j > i + THEILER) || (j < i - THEILER)) 
	        { 
                    distxij = 0;
	            for (k = 0; k < EMBED_X_TAU; k += TAU) 
	            {
                        xij = x[i + k] - x[j + k];
                        distxij += xij*xij;
	            }
                    distx[j] = distxij;

	            if (distxij < auxx[0]) 
		    {
	    	        for (k = 0; k < NN+1; k++) 
	    	        {
	    	            if (distxij < auxx[k])
                            {
                                auxx[k] = auxx[k+1];
                                indexx[k] = indexx[k+1];
                            }
			    else
			    {
			        auxx[k-1] = distxij;
			        indexx[k-1] = j;
			        break;			
		            }
		        }  
		    }

                    rrx += distxij;	
                    numPoints++;
	        }  
	    } 
            rr[ch] = rrx/numPoints;

        } 

        for(ch1 = 0; ch1 < channels; ch1++) {
            indexx = index[ch1]; 
            distx  = dist[ch1];
            auxx   = aux[ch1];
            rrx    = rr[ch1]; 
            for(ch2 = ch1+1; ch2 < channels; ch2++) {
                indexy = index[ch2]; 
                disty  = dist[ch2];
                auxy   = aux[ch2];
                rry    = rr[ch2];
 
	        /*rxx = ryy = rxy = ryx = 0; */
                rxy = ryx = 0;

	        for (k = 0; k < NN; k++)
	        {
                    /*rxx += auxx[k];
	            ryy += auxy[k]; */
	            rxy += distx[indexy[k]];
	            ryx += disty[indexx[k]];
	        }

	        /*rxx /= NN;								
	        ryy /= NN; */
	        rxy /= NN;
	        ryx /= NN;
                
                if ((rrx > 0) && (rry > 0))  
	             nxy[ch1][ch2] += (rrx - rxy)/rrx + (rry - ryx)/rry;
                else nxy[ch1][ch2] = 0;
	        /* nxy[ch2][ch1] += (rry - ryx)/(rry - ryy); */
            } /* ch2 loop ends */
        }  /* ch1 loop ends */
    } /* i loop ends */
   
    for(ch1 = 0; ch1 < channels; ch1++) 
    {     
        S[ch1][ch1] = 0;  
        for(ch2 = ch1 + 1; ch2 < channels; ch2++) 
        {
            S[ch1][ch2] = nxy[ch1][ch2]/(2*numDelayVectors);
            if (S[ch1][ch2] < 0)
                S[ch1][ch2] = 0;
            S[ch2][ch1] = S[ch1][ch2];    
        }  
    }

    free_dmatrix(nxy, channels);
    free_darray(rr);
    free_dmatrix(dist, channels);
    free_imatrix(index, channels);
    free_dmatrix(aux, channels);

    return;
}

double standardDeviation(double *x, int length)
{
    double xmean, xvar, xi;
    int i;
	
    xmean = xvar = 0; /* initialised to zero */

    for(i = 0; i < length; i++) /* summing x and y */
    {
        xmean += x[i];
    }

    xmean /= length;  /* average values */ 

    
    for(i = 0; i < length; i++)
    {
        xi = x[i] - xmean;
        xvar += xi*xi;
    }

    xvar = sqrt(xvar/(length-1)); /* variances */

    return xvar;
}


double *new_darray(int cols) {
   return (double *) mxCalloc(cols+1, sizeof(double));
}

void free_darray(double *array) {
   mxFree(array);
}

double **new_dmatrix(int rows, int cols) {
   double **matrix;
   int k; 

   matrix = (double **) mxCalloc(rows+1, sizeof(double *));
   for(k = 0; k < rows; k++)
      matrix[k] = new_darray(cols);

   return matrix;
}

void free_dmatrix(double **matrix, int rows) {
   int k;

   for(k = rows-1; k >= 0; k--)
      free_darray(matrix[k]);

   mxFree(matrix);
}

int *new_iarray(int cols) {
   return (int *) mxCalloc(cols+1, sizeof(int));
}

void free_iarray(int *array) {
   mxFree(array);
}

int **new_imatrix(int rows, int cols) {
   int **matrix;
   int k; 

   matrix = (int **) mxCalloc(rows+1, sizeof(int *));
   for(k = 0; k < rows; k++)
      matrix[k] = new_iarray(cols);

   return matrix;
}

void free_imatrix(int **matrix, int rows) {
   int k;

   for(k = rows-1; k >= 0; k--)
      free_iarray(matrix[k]);

   mxFree(matrix);
}

