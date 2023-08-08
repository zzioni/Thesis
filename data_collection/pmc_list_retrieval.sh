#!/bin/bash
esearch -db pmc -query "(\"deep learning\"[ABST] OR \"machine learning\"[ABST] OR \"artificial intelligence\"[ABST] OR \"computer vision\"[ABST] OR \"AI\"[ABST] OR \"ML\"[ABST] OR \"DL\"[ABST] OR \"deep learning\"[TITL] OR \"machine learning\"[TITL] OR \"artificial intelligence\"[TITL] OR \"computer vision\"[TITL] OR \"AI\"[TITL]  OR \"ML\"[TITL] OR \"DL\"[TITL] OR \"deep learning\"[MESH] OR \"machine learning\"[MESH] OR \"artificial intelligence\"[MESH] OR \"computer vision\"[MESH] OR \"AI\"[MESH] OR \"ML\"[MESH] OR \"DL\"[MESH] OR \"Artificial Neural Network\"[ABST] OR \"Artificial Neural Network\"[TITL] OR \"Artificial Neural Network\"[MESH]) AND (\"image\"[ABST] OR \"image\"[TITL] OR \"image\" [MESH] OR \"CT\"[ABST] OR \"CT\"[TITL] OR \"CT\" [MESH] OR \"imaging\"[ABST] OR \"imaging\"[TITL] OR \"imaging\" [MESH] OR \"radiography\"[ABST] OR \"radiography\"[TITL] OR \"radiography\" [MESH] OR \"X-ray\"[ABST] OR \"X-ray\"[TITL] OR \"X-ray\" [MESH] OR \"computed tomography\"[ABST] OR \"computed tomography\"[TITL] OR \"computed tomography\" [MESH] OR \"magnetic resonance imaging\"[ABST] OR \"magnetic resonance imaging\"[TITL] OR \"magnetic resonance\" [MESH] OR \"MRI\"[ABST] OR \"MRI\"[TITL] OR \"MRI\" [MESH] OR \"ultrasound\"[ABST] OR \"ultrasound\"[TITL] OR \"ultrasound\" [MESH] OR \"digital pathology\"[ABST] OR \"digital pathology\"[TITL] OR \"digital pathology\" [MESH] OR \"magnetic resonance tomography\"[ABST] OR \"magnetic resonance tomography\"[TITL] OR \"magnetic resonance tomography\"[MESH] OR \"MRT\"[ABST] OR \"MRT\"[TITL] OR \"MRT\"[MESH] OR \"positron emission tomography\"[ABST] OR \"positron emission tomography\"[TITL] OR \"positron emission tomography\"[MESH] OR \"PET\"[ABST] OR \"PET\"[TITL] OR \"PET\"[MESH] OR \"mammography\"[ABST] OR \"mammography\"[TITL] OR \"mammography\"[MESH] OR \"MG\"[ABST] OR \"MG\"[TITL] OR \"MG\"[MESH] OR \"magnetic resonance rngiography\"[ABST] OR \"magnetic resonance rngiography\"[TITL] OR \"mammographmagnetic resonance rngiography\"[MESH] OR \"MRA\"[ABST] OR \"MRA\"[TITL] OR \"MRA\"[MESH] OR \"nuclear medicine imaging\"[ABST] OR \"nuclear medicine imaging\"[TITL] OR \"nuclear medicine imaging\"[MESH]) AND (\"computer aided detection\"[ABST] OR \"computer aided detection\"[TITL] OR \"computer aided detection\"[MESH] OR \"computer aided diagnosis\"[ABST] OR \"computer aided diagnosis\"[TITL] OR \"computer aided diagnosis\"[MESH] OR \"CADe\"[ABST] OR \"CADe\"[TITL] OR \"CADe\"[MESH] OR \"CADx\"[ABST] OR \"CADx\"[TITL] OR \"CADx\"[MESH] OR \"CADx\"[ABST] OR \"CADx\"[TITL] OR \"CADx\"[MESH])" | efetch -format uilist > /Data/jiwon/pmc_list_TIAB/pmc_CAD_list