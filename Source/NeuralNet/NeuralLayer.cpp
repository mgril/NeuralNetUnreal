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