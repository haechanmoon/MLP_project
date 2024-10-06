#include "MLP.h"
#include <stdio.h>		// FILE�� ����
#include <malloc.h>		//malloc()�� ���� �߰�
#include <stdlib.h>		//srand()�������߰�
#include <time.h>		// time()������ �߰�
#include <math.h>		// exp()������ �߰�



CMLP::CMLP()
{
	m_iNumInNodes = 0;
	m_iNumOutNodes = 0;
	m_NumNodes = NULL;
	
	m_NodeOut = NULL;
	m_Weight = NULL;
	m_ErrorGradient = NULL;

	pInValue = NULL;
	pOutValue = NULL;
	pCorrectOutValue = NULL;

}

CMLP::~CMLP()
{
	int layer, snode;

	if (m_NodeOut != NULL)
	{
		for (layer = 0; layer < (m_iNumTotalLayer + 1); layer++)
			free(m_NodeOut[layer]);
		free(m_NodeOut);
	}

	if (m_Weight != NULL)
	{
		for (layer = 0; layer < (m_iNumTotalLayer - 1); layer++)
		{
			if (m_Weight[layer] != NULL)
			{
				for (snode = 0; snode < m_NumNodes[layer] + 1; snode++)
					free(m_Weight[layer][snode]);
				free(m_Weight[layer]);
			}
		}
		free(m_Weight);
	}

	if (m_ErrorGradient != NULL)
	{
		for (layer = 0; layer < (m_iNumTotalLayer); layer++)
			free(m_ErrorGradient[layer]);
		free(m_ErrorGradient);
	}

	if (m_NumNodes != NULL)
		free(m_NumNodes);

}

bool CMLP::Create(int InNode, int * pHiddenNode, int OutNode, int numHiddenLayer)
{
	int layer,snode, enode;

	m_iNumInNodes = InNode;
	m_iNumOutNodes = OutNode;
	m_iNumHiddenLayer = numHiddenLayer;// �Է�,����� ����
	m_iNumTotalLayer = numHiddenLayer + 2;		// ����+�Է�+���

	//m_iNumNodes�� ���� �޸��Ҵ�
	m_NumNodes = (int*)malloc((m_iNumTotalLayer + 1) * sizeof(int));	// ����(+1)

	m_NumNodes[0] = m_iNumInNodes;
	for (layer = 0; layer < m_iNumHiddenLayer; layer++)
		m_NumNodes[1 + layer] = pHiddenNode[layer];
	m_NumNodes[m_iNumTotalLayer - 1] = m_iNumOutNodes;	// ����� ����
	m_NumNodes[m_iNumTotalLayer] = m_iNumOutNodes;		// ����   ����

	// ����庰 ��¸޸��Ҵ�=[layerno][nodeno]
	// �Է�:m_NodeOut[0][],���m_NodeOut[m_iNumTotalLayer-1][]
	// ����:m_NodeOut[m_iNumTotalLayer][]
	m_NodeOut = (double**)malloc((m_iNumTotalLayer+1) * sizeof(double*));	// ����(+1)
	for (layer = 0; layer < m_iNumTotalLayer; layer++)
		m_NodeOut[layer] = (double*)malloc((m_NumNodes[layer]+1) * sizeof(double));	// ���̾�� ���� +1
	// ����(��� ���� ���� ����,���̾�� �ʿ������ ÷�ڴ� 1���� n����)
	m_NodeOut[m_iNumTotalLayer]= (double*)malloc((m_NumNodes[m_iNumTotalLayer-1]+1) * sizeof(double));

	// ����ġ �޸��Ҵ� m_Weight[����layer][���۳��][������]
	m_Weight = (double***)malloc((m_iNumTotalLayer - 1) * sizeof(double**));
	for (layer = 0; layer < m_iNumTotalLayer - 1; layer++)
	{
		m_Weight[layer] = (double**)malloc((m_NumNodes[layer] + 1) * sizeof(double*));	// ���̾(+1)
		for (snode = 0; snode < m_NumNodes[layer] + 1; snode++)
			m_Weight[layer][snode] = (double*)malloc((m_NumNodes[layer + 1]+1) * sizeof(double));	// �������̾��� ����
	}

	pInValue = m_NodeOut[0];
	pOutValue = m_NodeOut[m_iNumTotalLayer - 1];
	pCorrectOutValue = m_NodeOut[m_iNumTotalLayer];

	InitW();

	// ���̾�� ���� ��°�=1
	for (layer = 0; layer < m_iNumTotalLayer+1; layer++)
	{
		m_NodeOut[layer][0] = 1;
	}
	
	return true;
}


void CMLP::InitW()
{
	int layer, snode, enode;

	srand(time(NULL));

	for (layer = 0; layer < m_iNumTotalLayer - 1; layer++)		// ��������
	{
		for (snode = 0; snode <= m_NumNodes[layer]; snode++)	// for ���̾�� ���� 0����
		{
			for (enode = 1; enode <= m_NumNodes[layer + 1]; enode++)	// ���� ���̾��� ����
			{
				m_Weight[layer][snode][enode] = (double)rand() / RAND_MAX - 0.5;	// -0.5~0.5
			}
		}
	}
	
}


void CMLP::Forward()
{
	int layer, snode, enode;
	double wsum;	// ������

	for (layer = 0; layer < m_iNumTotalLayer - 1; layer++)
	{
		for (enode = 1; enode <= m_NumNodes[layer + 1]; enode++)
		{
			wsum = 0.0;			// ��庰 ������
			wsum += m_Weight[layer][0][enode] * 1;	//���̾ 
			for (snode = 1; snode <= m_NumNodes[layer]; snode++)
			{
				wsum += (m_Weight[layer][snode][enode] * m_NodeOut[layer][snode]);
			}
			
			m_NodeOut[layer + 1][enode] = ActivationFunc(wsum);
		}
	}

}


double CMLP::ActivationFunc(double weightsum)
{
	// step func
	//if (weightsum > 0)	return 1.0;
	//else		return 0.0;

	// sigmoid
	return 1 / (1 + exp(-weightsum));
}


void CMLP::BackPropagationLearning()
{
	int layer;

	// ������縦 ���� �޸� �Ҵ�
	if (m_ErrorGradient == NULL)
	{
		// ����庰 ��¸޸��Ҵ�=m_ErrorGrident[layerno][nodeno]
		// �Է�:m_ErrorGradient[0][],���m_ErrorGradient[m_iNumTotalLayer-1][]
		m_ErrorGradient = (double**)malloc((m_iNumTotalLayer) * sizeof(double*));	//
		for (layer = 0; layer < m_iNumTotalLayer; layer++)
			m_ErrorGradient[layer] = (double*)malloc((m_NumNodes[layer]+1) * sizeof(double));		// ���̾(0)�� ���� +1
	}

	int snode, enode, node;

	// �����error�����
	for (node = 1; node <= m_iNumOutNodes; node++)
	{
		m_ErrorGradient[m_iNumTotalLayer-1][node] =
			(pCorrectOutValue[node] - m_NodeOut[m_iNumTotalLayer - 1][node])
			*m_NodeOut[m_iNumTotalLayer -1][node] * (1 - m_NodeOut[m_iNumTotalLayer - 1][node]);
	}

	// error�����
	for (layer = m_iNumTotalLayer-2; layer >= 0; layer--)
	{
		for (snode = 1; snode <= m_NumNodes[layer]; snode++)
		{
			m_ErrorGradient[layer][snode] = 0.0;
			for (enode = 1; enode <= m_NumNodes[layer + 1]; enode++)
			{
				m_ErrorGradient[layer][snode] += (m_ErrorGradient[layer + 1][enode] * m_Weight[layer][snode][enode]);
			}
			m_ErrorGradient[layer][snode] *= m_NodeOut[layer][snode] * (1 - m_NodeOut[layer][snode]);
		}
	}

	// ����ġ����
	for (layer = m_iNumTotalLayer-2; layer >= 0; layer--)
	{
		for (enode = 1; enode <= m_NumNodes[layer + 1]; enode++)
		{
			m_Weight[layer][0][enode] += (LEARNING_RATE * m_ErrorGradient[layer + 1][enode] * 1);// ���̾
			for (snode = 1; snode <= m_NumNodes[layer]; snode++)
			{
				m_Weight[layer][snode][enode] += (LEARNING_RATE * m_ErrorGradient[layer + 1][enode] * m_NodeOut[layer][snode]);
			}
			
		}
	}

}


bool CMLP::SaveWeight(char* fname)
{
	int layer, snode, enode;
	FILE* fp;

	// Use fopen_s for safer file operations
	if (fopen_s(&fp, fname, "wt") != 0)
		return false;

	// �Է³��� ���緹�̾�� ��³���
	fprintf(fp, "%d %d %d\n", m_iNumInNodes, m_iNumHiddenLayer, m_iNumOutNodes);

	// node_layer0 node_layer1 node_layer2......
	for (layer = 0; layer < m_iNumTotalLayer; layer++)
	{
		fprintf(fp, "%d ", m_NumNodes[layer]);
	}
	fprintf(fp, "\n");

	// save weight
	for (layer = 0; layer < m_iNumTotalLayer - 1; layer++)
	{
		for (enode = 1; enode <= m_NumNodes[layer + 1]; enode++)
		{
			for (snode = 0; snode <= m_NumNodes[layer]; snode++) // ���̾������ 0����
			{
				fprintf(fp, "%.9lf ", m_Weight[layer][snode][enode]);
			}
		}
		fprintf(fp, "\n");
	}

	fclose(fp);

	return true;
}



bool CMLP::LoadWeight(char* fname)
{
	int layer, snode, enode;
	FILE* fp;

	// Use fopen_s for safer file operations
	if (fopen_s(&fp, fname, "rt") != 0)
		return false;

	// �Է³��� ���緹�̾�� ��³���
	fscanf_s(fp, "%d %d %d", &m_iNumInNodes, &m_iNumHiddenLayer, &m_iNumOutNodes);

	// node_layer0 node_layer1 node_layer2......
	for (layer = 0; layer < m_iNumTotalLayer; layer++)
	{
		fscanf_s(fp, "%d ", &m_NumNodes[layer]);
	}

	// load weight
	for (layer = 0; layer < m_iNumTotalLayer - 1; layer++)
	{
		for (enode = 1; enode <= m_NumNodes[layer + 1]; enode++)
		{
			for (snode = 0; snode <= m_NumNodes[layer]; snode++) // ���̾������ 0����
			{
				fscanf_s(fp, "%lf ", &m_Weight[layer][snode][enode]);
			}
		}
	}

	fclose(fp);

	return true;
}

int main()
{
	CMLP MultiLayer;

	int HiddenNodes[1] = { 2 };
	int numofHiddenLayer = 1;
	MultiLayer.Create(2, HiddenNodes, 1, numofHiddenLayer);

	double x[4][2] = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
	double correctOutputs[4] = { 0, 1, 1, 0 }; // XOR ���� ����

	// ����ġ �ʱ�ȭ (�������� ������ ��)
	MultiLayer.m_Weight[0][0][1] = -0.8;
	MultiLayer.m_Weight[0][1][1] = 0.5;
	MultiLayer.m_Weight[0][2][1] = 0.4;

	MultiLayer.m_Weight[0][0][2] = 0.1;
	MultiLayer.m_Weight[0][1][2] = 0.9;
	MultiLayer.m_Weight[0][2][2] = 1.0;

	MultiLayer.m_Weight[1][0][1] = -0.3;
	MultiLayer.m_Weight[1][1][1] = -1.2;
	MultiLayer.m_Weight[1][2][1] = 1.1;

	// �н� �ݺ�
	for (int epoch = 0; epoch < 1000; ++epoch) {
		for (int i = 0; i < 4; ++i) {
			// �Է°� ����
			MultiLayer.pInValue[1] = x[i][0];
			MultiLayer.pInValue[2] = x[i][1];

			// ���䰪 ����
			MultiLayer.pCorrectOutValue[1] = correctOutputs[i];

			// Forward ��� �� �н�
			MultiLayer.Forward();
			MultiLayer.BackPropagationLearning();
		}
	}

	// �н� �� ��� Ȯ��
	for (int i = 0; i < 4; ++i) {
		MultiLayer.pInValue[1] = x[i][0];
		MultiLayer.pInValue[2] = x[i][1];

		MultiLayer.Forward();

		printf("�Է�: [%lf, %lf] => ���: %lf (����: %lf)\n",
			x[i][0], x[i][1], MultiLayer.pOutValue[1], correctOutputs[i]);
	}

	return 0;
}

