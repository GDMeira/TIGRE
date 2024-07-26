/*-------------------------------------------------------------------------
 *
 * MATLAB MEX gateway for backprojection
 *
 * This file gets the data from MATLAB, checks it for errors and then
 * parses it to C and calls the relevant C/CUDA functions.
 *
 * CODE by Ander Biguri
 *
 * ---------------------------------------------------------------------------
 * ---------------------------------------------------------------------------
 * Copyright (c) 2015, University of Bath and CERN- European Organization for
 * Nuclear Research
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions de c�digo-fonte devem manter o aviso de direitos autorais acima,
 * esta lista de condi��es e a seguinte isen��o de responsabilidade.
 *
 * 2. Redistribui��es em formato bin�rio devem reproduzir o aviso de direitos autorais acima,
 * esta lista de condi��es e a seguinte isen��o de responsabilidade na documenta��o
 * e/ou outros materiais fornecidos com a distribui��o.
 *
 * 3. Nem o nome do detentor dos direitos autorais nem os nomes de seus colaboradores
 * podem ser usados para endossar ou promover produtos derivados deste software sem
 * permiss�o pr�via por escrito.
 *
 * ESTE SOFTWARE � FORNECIDO PELOS DETENTORES DOS DIREITOS AUTORAIS E COLABORADORES "COMO EST�"
 * E QUALQUER GARANTIA EXPRESSA OU IMPL�CITA, INCLUINDO, MAS N�O SE LIMITANDO �S,
 * GARANTIAS IMPL�CITAS DE COMERCIALIZA��O E ADEQUA��O A UM DETERMINADO PROP�SITO
 * S�O DECLINADAS. EM NENHUM EVENTO O DETENTOR DOS DIREITOS AUTORAIS OU COLABORADORES SER�O
 * RESPONS�VEIS POR QUALQUER DANO DIRETO, INDIRETO, INCIDENTAL, ESPECIAL, EXEMPLAR, OU
 * CONSEQUENCIAL (INCLUINDO, MAS N�O SE LIMITANDO �, AQUISI��O DE MERCADORIAS OU SERVI�OS;
 * PERDA DE USO, DADOS, OU LUCROS; OU INTERRUP��O DE NEG�CIOS) NO ENTANTO CAUSADO
 * E EM QUALQUER TEORIA DE RESPONSABILIDADE, SEJA EM CONTRATO, RESPONSABILIDADE ESTRITA,
 * OU DELITO (INCLUINDO NEGLIG�NCIA OU OUTRO) DECORRENTE DE QUALQUER FORMA DO USO DESTE
 * SOFTWARE, MESMO QUE AVISADO DA POSSIBILIDADE DE TAIS DANOS.
 * ---------------------------------------------------------------------------
 *
 * Contact: tigre.toolbox@gmail.com
 * Codes  : https://github.com/CERN/TIGRE
 * ---------------------------------------------------------------------------
 */

#include <math.h>
#include <string.h>
#include <tmwtypes.h>
#include <mex.h>
#include <matrix.h>
#include <CUDA/voxel_backprojection.hpp>
#include <CUDA/voxel_backprojection2.hpp>
#include <CUDA/voxel_backprojection_parallel.hpp>
#include <CUDA/GpuIds.hpp>

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[]) {
    // Check number of inputs
    if (nrhs != 5) {
        mexErrMsgIdAndTxt("CBCT:MEX:Atb:InvalidInput", "Wrong number of inputs provided");
    }

    // 5th argument is array of GPU-IDs.
    GpuIds gpuids;
    {
        size_t iM = mxGetM(prhs[4]);
        if (iM != 1) {
            mexErrMsgIdAndTxt("CBCT:MEX:Atb:unknown", "5th parameter must be a row vector.");
            return;
        }
        size_t uiGpuCount = mxGetN(prhs[4]);
        if (uiGpuCount == 0) {
            mexErrMsgIdAndTxt("CBCT:MEX:Atb:unknown", "5th parameter must be a row vector.");
            return;
        }
        int* piGpuIds = (int*)mxGetData(prhs[4]);
        gpuids.SetIds(uiGpuCount, piGpuIds);
    }

    // 4th argument is matched or unmatched.
    bool pseudo_matched = false; // Called krylov, because I designed it for krylov case....
    char* krylov = mxArrayToString(prhs[3]);
    if (!krylov) {
        mexErrMsgIdAndTxt("CBCT:MEX:Atb:InvalidInput", "4th parameter must be a string.");
    }
    if (!strcmp(krylov, "matched")) // if its 0, they are the same
        pseudo_matched = true;
    mxFree(krylov);

    // Third argument: angle of projection.
    size_t mrows, nangles;
    mrows = mxGetM(prhs[2]);
    nangles = mxGetN(prhs[2]);

    double const* anglesM = static_cast<double const*>(mxGetData(prhs[2]));
    float* angles = (float*)malloc(nangles * mrows * sizeof(float));
    if (angles == NULL) {
        mexErrMsgIdAndTxt("CBCT:MEX:Atb:MemoryError", "Memory allocation failed for angles.");
    }
    for (size_t i = 0; i < nangles * mrows; i++) {
        angles[i] = (float)anglesM[i];
    }

    // First input: The projections
    mxArray const* image = prhs[0];
    mwSize const numDims = mxGetNumberOfDimensions(image);
    if (!(numDims == 3 && nangles > 1) && !(numDims == 2 && nangles == 1)) {
        free(angles);
        mexErrMsgIdAndTxt("CBCT:MEX:Atb:InvalidInput", "Projection data is not the right size");
    }
    if (!mxIsSingle(prhs[0])) {
        free(angles);
        mexErrMsgIdAndTxt("CBCT:MEX:Ax:InvalidInput", "Input image must be a single noncomplex array.");
    }

    float* projections = static_cast<float*>(mxGetData(image));

    // Second input: Geometry structure
    mxArray* geometryMex = (mxArray*)prhs[1];
    const char* fieldnames[] = {
        "nVoxel", "sVoxel", "dVoxel", "nDetector", "sDetector", "dDetector", "DSD", "DSO", "offOrigin",
        "offDetector", "accuracy", "mode", "COR", "rotDetector", "EPS", "gelTubeRadius", "nWater", "nGel"
    };

    double* nVoxel, * nDetec;
    double* sVoxel, * dVoxel, * sDetec, * dDetec, * DSO, * DSD, * offOrig, * offDetec, * EPS;
    double* acc, * COR, * rotDetector, * gelTubeRadius, * nWater, * nGel;
    const char* mode;
    bool coneBeam = true;
    Geometry geo;
    int c;
    geo.unitX = 1; geo.unitY = 1; geo.unitZ = 1;

    for (int ifield = 0; ifield < 18; ifield++) {
        mxArray* tmp = mxGetField(geometryMex, 0, fieldnames[ifield]);
        if (tmp == NULL) {
            mexErrMsgIdAndTxt("CBCT:MEX:Atb:InvalidInput", "Missing field in geometry struct.");
        }
        switch (ifield) {
        case 0:
            nVoxel = (double*)mxGetData(tmp);
            geo.nVoxelX = (int)nVoxel[0];
            geo.nVoxelY = (int)nVoxel[1];
            geo.nVoxelZ = (int)nVoxel[2];
            break;
        case 1:
            sVoxel = (double*)mxGetData(tmp);
            geo.sVoxelX = (float)sVoxel[0];
            geo.sVoxelY = (float)sVoxel[1];
            geo.sVoxelZ = (float)sVoxel[2];
            break;
        case 2:
            dVoxel = (double*)mxGetData(tmp);
            geo.dVoxelX = (float)dVoxel[0];
            geo.dVoxelY = (float)dVoxel[1];
            geo.dVoxelZ = (float)dVoxel[2];
            break;
        case 3:
            nDetec = (double*)mxGetData(tmp);
            geo.nDetecU = (int)nDetec[0];
            geo.nDetecV = (int)nDetec[1];
            break;
        case 4:
            sDetec = (double*)mxGetData(tmp);
            geo.sDetecU = (float)sDetec[0];
            geo.sDetecV = (float)sDetec[1];
            break;
        case 5:
            dDetec = (double*)mxGetData(tmp);
            geo.dDetecU = (float)dDetec[0];
            geo.dDetecV = (float)dDetec[1];
            break;
        case 6:
            geo.DSD = (float*)malloc(nangles * sizeof(float));
            if (geo.DSD == NULL) {
                free(angles);
                mexErrMsgIdAndTxt("CBCT:MEX:Atb:MemoryError", "Memory allocation failed for DSD.");
            }
            DSD = (double*)mxGetData(tmp);
            for (size_t i = 0; i < nangles; i++) {
                geo.DSD[i] = (float)DSD[i];
            }
            break;
        case 7:
            geo.DSO = (float*)malloc(nangles * sizeof(float));
            if (geo.DSO == NULL) {
                free(angles);
                free(geo.DSD);
                mexErrMsgIdAndTxt("CBCT:MEX:Atb:MemoryError", "Memory allocation failed for DSO.");
            }
            DSO = (double*)mxGetData(tmp);
            for (size_t i = 0; i < nangles; i++) {
                geo.DSO[i] = (float)DSO[i];
            }
            break;
        case 8:
            geo.offOrigX = (float*)malloc(nangles * sizeof(float));
            geo.offOrigY = (float*)malloc(nangles * sizeof(float));
            geo.offOrigZ = (float*)malloc(nangles * sizeof(float));
            if (geo.offOrigX == NULL || geo.offOrigY == NULL || geo.offOrigZ == NULL) {
                free(angles);
                free(geo.DSD);
                free(geo.DSO);
                mexErrMsgIdAndTxt("CBCT:MEX:Atb:MemoryError", "Memory allocation failed for offOrig.");
            }
            offOrig = (double*)mxGetData(tmp);
            for (size_t i = 0; i < nangles; i++) {
                geo.offOrigX[i] = (float)offOrig[0 + 3 * i];
                geo.offOrigY[i] = (float)offOrig[1 + 3 * i];
                geo.offOrigZ[i] = (float)offOrig[2 + 3 * i];
            }
            break;
        case 9:
            geo.offDetecU = (float*)malloc(nangles * sizeof(float));
            geo.offDetecV = (float*)malloc(nangles * sizeof(float));
            if (geo.offDetecU == NULL || geo.offDetecV == NULL) {
                free(angles);
                free(geo.DSD);
                free(geo.DSO);
                free(geo.offOrigX);
                free(geo.offOrigY);
                free(geo.offOrigZ);
                mexErrMsgIdAndTxt("CBCT:MEX:Atb:MemoryError", "Memory allocation failed for offDetec.");
            }
            offDetec = (double*)mxGetData(tmp);
            for (size_t i = 0; i < nangles; i++) {
                geo.offDetecU[i] = (float)offDetec[0 + 2 * i];
                geo.offDetecV[i] = (float)offDetec[1 + 2 * i];
            }
            break;
        case 10:
            acc = (double*)mxGetData(tmp);
            geo.accuracy = (float)acc[0];
            if (acc[0] < 0.001) {
                free(angles);
                free(geo.DSD);
                free(geo.DSO);
                free(geo.offOrigX);
                free(geo.offOrigY);
                free(geo.offOrigZ);
                free(geo.offDetecU);
                free(geo.offDetecV);
                mexErrMsgIdAndTxt("CBCT:MEX:Ax:Accuracy", "Accuracy should be bigger than 0.001");
            }
            break;
        case 11:
            mode = mxArrayToString(tmp);
            if (!mode) {
                free(angles);
                free(geo.DSD);
                free(geo.DSO);
                free(geo.offOrigX);
                free(geo.offOrigY);
                free(geo.offOrigZ);
                free(geo.offDetecU);
                free(geo.offDetecV);
                mexErrMsgIdAndTxt("CBCT:MEX:Atb:InvalidInput", "Mode field must be a string.");
            }
            if (!strcmp(mode, "parallel"))
                coneBeam = false;
            mxFree((void*)mode);
            break;
        case 12:
            COR = (double*)mxGetData(tmp);
            geo.COR = (float*)malloc(nangles * sizeof(float));
            if (geo.COR == NULL) {
                free(angles);
                free(geo.DSD);
                free(geo.DSO);
                free(geo.offOrigX);
                free(geo.offOrigY);
                free(geo.offOrigZ);
                free(geo.offDetecU);
                free(geo.offDetecV);
                mexErrMsgIdAndTxt("CBCT:MEX:Atb:MemoryError", "Memory allocation failed for COR.");
            }
            for (size_t i = 0; i < nangles; i++) {
                geo.COR[i] = (float)COR[i];
            }
            break;
        case 13:
            geo.dRoll = (float*)malloc(nangles * sizeof(float));
            geo.dPitch = (float*)malloc(nangles * sizeof(float));
            geo.dYaw = (float*)malloc(nangles * sizeof(float));
            if (geo.dRoll == NULL || geo.dPitch == NULL || geo.dYaw == NULL) {
                free(angles);
                free(geo.DSD);
                free(geo.DSO);
                free(geo.offOrigX);
                free(geo.offOrigY);
                free(geo.offOrigZ);
                free(geo.offDetecU);
                free(geo.offDetecV);
                free(geo.COR);
                mexErrMsgIdAndTxt("CBCT:MEX:Atb:MemoryError", "Memory allocation failed for rotation angles.");
            }
            rotDetector = (double*)mxGetData(tmp);
            for (size_t i = 0; i < nangles; i++) {
                geo.dYaw[i] = (float)rotDetector[0 + 3 * i];
                geo.dPitch[i] = (float)rotDetector[1 + 3 * i];
                geo.dRoll[i] = (float)rotDetector[2 + 3 * i];
            }
            break;
        case 14:
            EPS = (double*)mxGetData(tmp);
            geo.EPS = (float)EPS[0];
            break;
        case 15:
            gelTubeRadius = (double*)mxGetData(tmp);
            geo.gelTubeRadius = (float)gelTubeRadius[0];
            break;
        case 16:
            nWater = (double*)mxGetData(tmp);
            geo.nWater = (float)nWater[0];
            break;
        case 17:
            nGel = (double*)mxGetData(tmp);
            geo.nGel = (float)nGel[0];
            break;
        default:
            mexErrMsgIdAndTxt("CBCT:MEX:Atb:unknown", "This should not happen. Weird");
            break;
        }
    }

    // Output allocation
    mwSize imgsize[3] = { geo.nVoxelX, geo.nVoxelY, geo.nVoxelZ };
    plhs[0] = mxCreateNumericArray(3, imgsize, mxSINGLE_CLASS, mxREAL);
    float* result = (float*)mxGetData(plhs[0]);

    // Call the CUDA kernel
    if (coneBeam) {
        if (pseudo_matched) {
            voxel_backprojection2(projections, geo, result, angles, nangles, gpuids);
        }
        else {
            voxel_backprojection(projections, geo, result, angles, nangles, gpuids);
        }
    }
    else {
        voxel_backprojection_parallel(projections, geo, result, angles, nangles, gpuids);
    }

    // Free allocated memory
    free(angles);
    free(geo.DSD);
    free(geo.DSO);
    free(geo.offOrigX);
    free(geo.offOrigY);
    free(geo.offOrigZ);
    free(geo.offDetecU);
    free(geo.offDetecV);
    free(geo.COR);
    free(geo.dRoll);
    free(geo.dPitch);
    free(geo.dYaw);
    free(result);
}
