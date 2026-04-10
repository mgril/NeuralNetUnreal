// Fill out your copyright notice in the Description page of Project Settings.


#include "NeuralLayer.h"

NeuralLayer::NeuralLayer()
{
}

NeuralLayer::~NeuralLayer()
{
}

void FNeuralLayer::Initialize(int32 InNumInputs, int32 InNumNeurons)
{
    NumInputs = InNumInputs;
    NumNeurons = InNumNeurons;

    // Allocation
    Weights.SetNum(NumInputs * NumNeurons);
    Biases.SetNum(NumNeurons);
    Outputs.SetNum(NumNeurons);
    WeightedInputs.SetNum(NumNeurons);
    NodeValues.SetNum(NumNeurons);

    // Poids aleatoires entre -1 et 1
    for (float& W : Weights)
    {
        W = FMath::RandRange(-1.0f, 1.0f);
    }

    // Biais a 0
    for (float& B : Biases)
    {
        B = 0.0f;
    }
}

TArray<float> FNeuralLayer::ForwardPass(const TArray<float>& Inputs)
{
    // Securite : verifier que les inputs ont la bonne taille
    check(Inputs.Num() == NumInputs);

    for (int32 j = 0; j < NumNeurons; j++)
    {
        // Somme (Input_i * Weight_ij) + Bias_j
        float WeightedSum = Biases[j];

        for (int32 i = 0; i < NumInputs; i++)
        {
            WeightedSum += Inputs[i] * GetWeight(j, i);
        }

        WeightedInputs[j] = WeightedSum;           // Stocke pour la backprop
        Outputs[j] = Sigmoid(WeightedSum);  // Activation
    }

    return Outputs;
}
// ─── Output Layer ─────────────────────────────────────────────────────────────
// NodeValue = 2*(Output - Expected) * SigmoidDerivative(WeightedInput)
// C'est dC/dO * dO/dWI — page 41-42 du PDF
void FNeuralLayer::ComputeOutputLayerNodeValues(const TArray<float>& Expected)
{
    check(Expected.Num() == NumNeurons);

    for (int32 j = 0; j < NumNeurons; j++)
    {
        float dC_dO = 2.0f * (Outputs[j] - Expected[j]);
        float dO_dWI = SigmoidDerivative(WeightedInputs[j]);

        NodeValues[j] = dC_dO * dO_dWI;
    }
}

// ─── Hidden Layer ─────────────────────────────────────────────────────────────
// NodeValue = Σ(NextWeight * NextNodeValue) * SigmoidDerivative(WeightedInput)
// On cumule l'influence de chaque connexion sortante — page 53-54 de ton PDF

void FNeuralLayer::ComputeHiddenLayerNodeValues(const FNeuralLayer& NextLayer)
{
    for (int32 j = 0; j < NumNeurons; j++)
    {
        // Cumul de l'influence de ce neurone sur tous les neurones du layer suivant
        float dC_dI = 0.0f;

        for (int32 k = 0; k < NextLayer.NumNeurons; k++)
        {
            // Le poids de la connexion j→k dans le layer suivant
            dC_dI += NextLayer.NodeValues[k] * NextLayer.GetWeight(k, j);
        }

        float dI_dWI = SigmoidDerivative(WeightedInputs[j]);
        NodeValues[j] = dC_dI * dI_dWI;  
    }
}
// ─── Apply Gradients ──────────────────────────────────────────────────────────
// Weight -= LearningRate * NodeValue * ConnectionInput
// Bias   -= LearningRate * NodeValue
// Page 57-58 du PDF

void FNeuralLayer::ApplyGradients(const TArray<float>& Inputs, float LearningRate)
{
    check(Inputs.Num() == NumInputs);

    for (int32 j = 0; j < NumNeurons; j++)
    {
        // Mise a jour du biais
        Biases[j] -= LearningRate * NodeValues[j];

        // Mise a jour de chaque poids entrant vers ce neurone
        for (int32 i = 0; i < NumInputs; i++)
        {
            Weights[j * NumInputs + i] -= LearningRate * NodeValues[j] * Inputs[i];
        }
    }
}
// ─── Helpers ─────────────────────────────────────────────────────────

float FNeuralLayer::GetWeight(int32 NeuronIndex, int32 InputIndex) const
{
    return Weights[NeuronIndex * NumInputs + InputIndex];
}



float FNeuralLayer::Sigmoid(float X)
{
    return 1.0f / (1.0f + FMath::Exp(-X));
}

float FNeuralLayer::SigmoidDerivative(float X)
{
    // Derivee de sigmoid : sigmoid(x) * (1 - sigmoid(x))
    float S = Sigmoid(X);
    return S * (1.0f - S);
}