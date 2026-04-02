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
    // Il faut au moins un input layer et un output layer
    check(LayerSizes.Num() >= 2);

    Layers.Empty();

    // On cree un layer entre chaque paire de tailles
    // LayerSizes[0] = nb d'inputs
    // LayerSizes[1] = nb de neurones du premier layer, etc.
    for (int32 i = 1; i < LayerSizes.Num(); i++)
    {
        FNeuralLayer Layer;
        Layer.Initialize(LayerSizes[i - 1], LayerSizes[i]);
        Layers.Add(Layer);
    }
}

TArray<float> FNeuralNetwork::Predict(const TArray<float>& Inputs)
{
    // Le premier layer recoit les inputs bruts
    TArray<float> CurrentInputs = Inputs;

    // Chaque layer recoit les outputs du layer precedent
    for (FNeuralLayer& Layer : Layers)
    {
        CurrentInputs = Layer.ForwardPass(CurrentInputs);
    }

    // CurrentInputs contient maintenant les outputs du dernier layer
    return CurrentInputs;
}

float FNeuralNetwork::ComputeCost(const TArray<float>& Output,
    const TArray<float>& Expected)
{
    check(Output.Num() == Expected.Num());

    // somme (Expected - Output)^2   formule page 25 du PDF
    float TotalCost = 0.0f;

    for (int32 i = 0; i < Output.Num(); i++)
    {
        float Diff = Expected[i] - Output[i];
        TotalCost += Diff * Diff;
    }

    return TotalCost;
}