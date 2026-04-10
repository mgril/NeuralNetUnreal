// Fill out your copyright notice in the Description page of Project Settings.


#include "NeuralNetworkData.h"

void UNeuralNetworkData::SaveFromNetwork(const FNeuralNetwork& Network)
{
    Layers.Empty();
    LearningRate = Network.LearningRate;

    for (const FNeuralLayer& Layer : Network.Layers)
    {
        FLayerData Data;
        Data.NumInputs = Layer.NumInputs;
        Data.NumNeurons = Layer.NumNeurons;
        Data.Weights = Layer.Weights;
        Data.Biases = Layer.Biases;
        Layers.Add(Data);
    }

    // Marque l'asset comme modifié pour qu'Unreal propose de sauvegarder
#if WITH_EDITOR
    MarkPackageDirty();
#endif
}

bool UNeuralNetworkData::LoadIntoNetwork(FNeuralNetwork& OutNetwork) const
{
    if (Layers.Num() == 0)
    {
        UE_LOG(LogTemp, Error, TEXT("NeuralNetworkData: asset vide, rien a charger."));
        return false;
    }

    OutNetwork.Layers.Empty();
    OutNetwork.LearningRate = LearningRate;

    for (const FLayerData& Data : Layers)
    {
        FNeuralLayer Layer;
        Layer.NumInputs = Data.NumInputs;
        Layer.NumNeurons = Data.NumNeurons;
        Layer.Weights = Data.Weights;
        Layer.Biases = Data.Biases;

        // Alloue les tableaux intermediaires
        Layer.Outputs.SetNum(Data.NumNeurons);
        Layer.WeightedInputs.SetNum(Data.NumNeurons);
        Layer.NodeValues.SetNum(Data.NumNeurons);

        OutNetwork.Layers.Add(Layer);
    }

    return true;
}