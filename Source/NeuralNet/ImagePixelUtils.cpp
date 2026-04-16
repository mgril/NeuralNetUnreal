// Fill out your copyright notice in the Description page of Project Settings.



#include "ImagePixelUtils.h"
#include "AssetRegistry/AssetRegistryModule.h"

bool FImagePixelUtils::TextureToFloatArray(UTexture2D* Texture,
    TArray<float>& OutPixels,
    int32 TargetSize)
{
    if (!Texture) return false;

    // Force le chargement complet de la texture en RAM
    Texture->SetForceMipLevelsToBeResident(30.0f);
    Texture->WaitForStreaming();


    // Vérifie que les données sont bien disponibles
    if (!Texture->GetPlatformData() ||
        Texture->GetPlatformData()->Mips.Num() == 0)
    {
        UE_LOG(LogTemp, Error, TEXT("Texture sans données : %s"),
            *Texture->GetName());
        return false;
    }

    // Force la lecture des pixels en RAM
    FTexture2DMipMap& Mip = Texture->GetPlatformData()->Mips[0];

    // Force le chargement du mip en RAM si pas encore fait
    if (!Mip.BulkData.IsBulkDataLoaded())
    {
        Mip.BulkData.ForceBulkDataResident();
    }
    FByteBulkData& RawData = Mip.BulkData;
    const uint8* RawBytes = static_cast<const uint8*>(RawData.Lock(LOCK_READ_ONLY));
    if (!RawBytes)
    {
        RawData.Unlock();
        UE_LOG(LogTemp, Error, TEXT("RawBytes null pour : %s"),
            *Texture->GetName());
        return false;
    }

    int32 Width = Mip.SizeX;
    int32 Height = Mip.SizeY;

    OutPixels.Empty();
    OutPixels.SetNum(TargetSize * TargetSize);

    // Lit chaque pixel et normalise entre 0 et 1
    // Format BGRA : 4 bytes par pixel
    for (int32 y = 0; y < TargetSize; y++)
    {
        for (int32 x = 0; x < TargetSize; x++)
        {
            // Redimensionne si necessaire
            int32 SrcX = FMath::Clamp((x * Width) / TargetSize, 0, Width - 1);
            int32 SrcY = FMath::Clamp((y * Height) / TargetSize, 0, Height - 1);

            int32 PixelIndex = (SrcY * Width + SrcX) * 4;

            // Convertit BGR en niveaux de gris
            float B = RawBytes[PixelIndex + 0] / 255.0f;
            float G = RawBytes[PixelIndex + 1] / 255.0f;
            float R = RawBytes[PixelIndex + 2] / 255.0f;

            // Formule standard RGB  Grayscale
            float Gray = 0.299f * R + 0.587f * G + 0.114f * B;

            OutPixels[y * TargetSize + x] = Gray;
        }
    }

    RawData.Unlock();
    return true;
}

void FImagePixelUtils::LoadDataset(const FString& FolderPath,
    TArray<TArray<float>>& OutInputs,
    TArray<TArray<float>>& OutLabels,
    int32 NumClasses,
    int32 TargetSize)
{
    OutInputs.Empty();
    OutLabels.Empty();

    FAssetRegistryModule& Registry =
        FModuleManager::LoadModuleChecked<FAssetRegistryModule>("AssetRegistry");

    // Parcourt chaque classe (sous-dossier 0 a 9)
    for (int32 ClassIndex = 0; ClassIndex < NumClasses; ClassIndex++)
    {
        FString ClassFolder = FString::Printf(TEXT("%s/%d"), *FolderPath, ClassIndex);

        TArray<FAssetData> Assets;
        Registry.Get().GetAssetsByPath(FName(*ClassFolder), Assets, false);

        UE_LOG(LogTemp, Warning,
            TEXT("Classe %d : %d images trouvées"), ClassIndex, Assets.Num());

        for (FAssetData& Asset : Assets)
        {
            UTexture2D* Texture = Cast<UTexture2D>(Asset.GetAsset());
            if (!Texture) continue;

           // Lecture des pixels
            TArray<float> Pixels;
            if (!TextureToFloatArray(Texture, Pixels, TargetSize)) continue;

            // Label en one-hot encoding
            // ex: classe 3  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
            TArray<float> Label;
            Label.SetNum(NumClasses);
            for (int32 i = 0; i < NumClasses; i++)
                Label[i] = (i == ClassIndex) ? 1.0f : 0.0f;

            OutInputs.Add(Pixels);
            OutLabels.Add(Label);
        }
    }

    UE_LOG(LogTemp, Warning,
        TEXT("Dataset charge : %d images au total"), OutInputs.Num());
}