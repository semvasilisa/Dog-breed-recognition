# Dog-breed-recognition

Projekt obejmował opracowanie i wytrenowanie sieci neuronowej, która rozpoznaje rasy psów na podstawie zdjęć. Wykorzystano zbiór danych Stanford Dogs Dataset, zawierający 120 ras psów i około 20 000 zdjęć.

## Struktura projektu
Projekt składa się z trzech głównych plików:

### 1. `createdCNN.py`
Ten plik zawiera utworzoną od podstaw splotową sieć neuronową.

W tej części projektu:
- Zaprojektowano i zaimplementowano prostą architekturę sieci neuronowych (CNN)
- Celem było zrozumienie działania splotu, łączenia, spłaszczania i gęstych warstw
- Model został wytrenowany na 10 rasach psów

Ta wersja pomogła zrozumieć wewnętrzną strukturę sieci neuronowych (CNN), ale model był zbyt prosty, aby osiągnąć wysoką dokładność w tak złożonym zbiorze danych.

---

### 2. `retraineddog.py` 
To jest główna i ostateczna wersja projektu.

Wykorzystuje ona wstępnie wytrenowany model EfficientNetB0, trenowany na zbiorze danych ImageNet.

W tym pliku:
- EfficientNetB0 jest używany jako ekstraktor cech
- Przetestowano użycie Fine-tuning
- Wytrenowano i oceniono ostateczny model
- Przeanalizowano wydajność modelu
- Utworzono wizualizacje
- Przeanalizowano błędy i błędne klasyfikacje

---

### 3. `resnet50dog.py`
Ten plik zawiera alternatywny model oparty na ResNet50, innym wstępnie wytrenowanym modelu.

W tej części:
- Użyto ResNet50 zamiast EfficientNet
- Model wytrenowano na tym samym zbiorze danych
- Wyniki porównano z EfficientNet

Ten eksperyment wykazał, że EfficientNet działał lepiej niż ResNet50 w tym zadaniu.

---

## Zastosowane technologie

- Python  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  
- Wstępnie wytrenowane modele: EfficientNetB0, ResNet50  

---

## Wyniki

- Stworzona CNN działała, ale była zbyt słaba dla tego zbioru danych
- Z wykorzystaniem EfficientNetB0 sieć osiągnęła najwyższą dokładność
- Fine-tuning nie poprawił wyników
- ResNet50 wypadł gorzej niż EfficientNet
