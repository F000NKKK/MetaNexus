using MetaNexus.Lib.NeuralNetwork.Tensors;
using System;

namespace MetaNexus.Lib.NeuralNetwork.ML.Layers.Abstractions
{
    public abstract class Layer
    {
        private Tensor biases;
        private Tensor weights;

        public int Size { get; set; }
        public int InputSize { get; set; }

        private static Random random = new Random(DateTime.Now.Millisecond);

        // Конструктор с размером входа и размером слоя
        public Layer(int inputSize, int size)
        {
            if (size <= 0)
                throw new ArgumentException("Размер слоя должен быть положительным числом.");

            Size = size;
            InputSize = inputSize;

            InitializeWeightsAndBiases();
        }

        // Конструктор с размером слоя (если входной размер равен размеру слоя)
        public Layer(int inputSize)
        {
            if (inputSize <= 0)
                throw new ArgumentException("Размер слоя должен быть положительным числом.");

            Size = inputSize;
            InputSize = inputSize;

            InitializeWeightsAndBiases();
        }

        // Конструктор для инициализации с существующими весами и смещениями
        public Layer(int inputSize, int size, Tensor weights, Tensor biases)
        {
            if (weights.Shape[0] != inputSize || weights.Shape[1] != size)
                throw new ArgumentException("Размеры тензоров весов не совпадают с размером слоя.");

            if (biases.Shape[0] != size)
                throw new ArgumentException("Размер тензора смещений не совпадает с размером слоя.");

            Size = size;
            InputSize = inputSize;
            this.weights = weights;
            this.biases = biases;
        }

        // Абстрактный метод для выполнения прямого прохода
        public abstract Tensor Forward(Tensor input);

        // Функции для получения и установки значений весов и смещений
        public Tensor GetWeights() => weights;

        public void SetWeights(Tensor newWeights)
        {
            if (newWeights.Shape[0] != InputSize || newWeights.Shape[1] != Size)
                throw new ArgumentException("Размеры нового тензора весов не совпадают с размерами слоя.");

            weights = newWeights;
        }

        public Tensor GetBiases() => biases;

        public void SetBiases(Tensor newBiases)
        {
            if (newBiases.Shape[0] != Size)
                throw new ArgumentException("Размеры нового тензора смещений не совпадают с размерами слоя.");

            biases = newBiases;
        }

        // Инициализация весов и смещений случайными значениями
        private void InitializeWeightsAndBiases()
        {
            weights = new Tensor(new int[] { InputSize, Size });
            biases = new Tensor(new int[] { Size });

            // Инициализация весов случайными значениями
            for (int i = 0; i < InputSize; i++)
            {
                for (int j = 0; j < Size; j++)
                {
                    weights[i, j] = (float)(random.NextDouble() * 2.0 - 1.0);
                }
            }

            // Инициализация смещений случайными значениями
            for (int i = 0; i < Size; i++)
            {
                biases[i] = (float)(random.NextDouble() * 2.0 - 1.0);
            }
        }
    }
}
