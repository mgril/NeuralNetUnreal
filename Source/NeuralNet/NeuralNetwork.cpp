// Fill out your copyright notice in the Description page of Project Settings.


#include "NeuralNetwork.h"

FNeuralNetwork::FNeuralNetwork()
{
}

FNeuralNetwork::~FNeuralNetwork()
{
}

void FNeuralNetwork::Build(const TArray<int32>& LayerSizes)
{
    check(LayerSizes.Num() >= 2);

    Layers.Empty();


    for (int32 i = 1; i < LayerSizes.Num(); i++)
    {
        FNeuralLayer Layer;
        Layer.Initialize(LayerSizes[i - 1], LayerSizes[i]);
        Layers.Add(Layer);
    }
}

TArray<float> FNeuralNetwork::Predict(const TArray<float>& Inputs)
{
    TArray<float> CurrentInputs = Inputs;

    // Layer recoit les outputs du layer precedent
    for (FNeuralLayer& Layer : Layers)
    {
        CurrentInputs = Layer.ForwardPass(CurrentInputs);
    }

    // CurrentInputs outputs du dernier layer
    return CurrentInputs;
}

float FNeuralNetwork::ComputeCost(const TArray<float>& Output,
    const TArray<float>& Expected)
{
    check(Output.Num() == Expected.Num());

    float TotalCost = 0.0f;

    for (int32 i = 0; i < Output.Num(); i++)
    {
        float Diff = Expected[i] - Output[i];
        TotalCost += Diff * Diff;
    }

    return TotalCost;
}

float FNeuralNetwork::Train(const TArray<float>& Inputs,
    const TArray<float>& Expected)
{
    TArray<float> Output = Predict(Inputs);

    Backpropagate(Inputs, Expected);

    return ComputeCost(Output, Expected);
}

void FNeuralNetwork::Backpropagate(const TArray<float>& Inputs,
    const TArray<float>& Expected)
{
    int32 LastIndex = Layers.Num() - 1;

    Layers[LastIndex].ComputeOutputLayerNodeValues(Expected);

    //NodeValues des hidden layers on remonte
    for (int32 i = LastIndex - 1; i >= 0; i--)
    {
        Layers[i].ComputeHiddenLayerNodeValues(Layers[i + 1]);
    }

    // Le premier layer recoit les inputs bruts les suivants recoivent les outputs du layer precedent
    for (int32 i = 0; i < Layers.Num(); i++)
    {
        const TArray<float>& LayerInputs = (i == 0) ? Inputs : Layers[i - 1].Outputs;

        Layers[i].ApplyGradients(LayerInputs, LearningRate);
    }
}