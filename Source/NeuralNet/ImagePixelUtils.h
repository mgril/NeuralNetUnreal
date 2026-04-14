// Fill out your copyright notice in the Description page of Project Settings.

#pragma once
#include "CoreMinimal.h"
#include "Engine/Texture2D.h"

class NEURALNET_API FImagePixelUtils
{
public:
    // Lit les pixels d'une texture  TArray<float> entre 0 et 1
    static bool TextureToFloatArray(UTexture2D* Texture,
        TArray<float>& OutPixels,
        int32 TargetSize = 28);

    // Charge toutes les textures d'un dossier avec leur label
    static void LoadDataset(const FString& FolderPath,
        TArray<TArray<float>>& OutInputs,
        TArray<TArray<float>>& OutLabels,
        int32 NumClasses,
        int32 TargetSize = 28);
};
