using MetaNexus.Lib.NeuralNetwork.Tensors;

namespace MetaNexus.Lib.NeuralNetwork.ML.Layers.Abstractions
{
    /// <summary>
    /// Абстрактный класс для слоя нейронной сети.
    /// </summary>
    public abstract class Layer : ILayer
    {
        /// <summary>
        /// Конструктор для создания слоя с заданными размерами.
        /// </summary>
        /// <param name="inputSize">Размер входа.</param>
        /// <param name="size">Размер слоя (количество нейронов).</param>
        public Layer(int inputSize, int size)
        {
            if (size <= 0)
                throw new ArgumentException("Размер слоя должен быть положительным числом.");

            Size = size;
            InputSize = inputSize;

            ((ILayer)this).InitializeWeightsAndBiases();
        }

        /// <summary>
        /// Конструктор для создания слоя с размером входа равным размеру слоя.
        /// </summary>
        /// <param name="inputSize">Размер входа.</param>
        public Layer(int inputSize)
        {
            if (inputSize <= 0)
                throw new ArgumentException("Размер слоя должен быть положительным числом.");

            Size = inputSize;
            InputSize = inputSize;

            ((ILayer)this).InitializeWeightsAndBiases();
        }

        /// <summary>
        /// Конструктор для создания слоя с заданными весами и смещениями.
        /// </summary>
        /// <param name="inputSize">Размер входа.</param>
        /// <param name="size">Размер слоя (количество нейронов).</param>
        /// <param name="weights">Тензор весов для слоя.</param>
        /// <param name="biases">Тензор смещений для слоя.</param>
        public Layer(int inputSize, int size, Tensor weights, Tensor biases)
        {
            if (weights.Shape[0] != inputSize || weights.Shape[1] != size)
                throw new ArgumentException("Размеры тензоров весов не совпадают с размером слоя.");

            if (biases.Shape[0] != size)
                throw new ArgumentException("Размеры тензора смещений не совпадают с размером слоя.");

            Size = size;
            InputSize = inputSize;
            this.weights = weights;
            this.biases = biases;
        }

        void ILayer.InitializeWeightsAndBiases()
        {
            weights = new Tensor(new int[] { InputSize, Size });
            biases = new Tensor(new int[] { Size });

            for (int i = 0; i < InputSize; i++)
            {
                for (int j = 0; j < Size; j++)
                {
                    weights[i, j] = (float)(random.NextDouble() * 2.0 - 1.0);
                }
            }

            for (int i = 0; i < Size; i++)
            {
                biases[i] = (float)(random.NextDouble() * 2.0 - 1.0);
            }
        }

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

        public abstract Tensor Forward(Tensor input);

        public Tensor Backward(Tensor delta, float learningRate)
        {
            // 1. Градиенты для весов и смещений
            Tensor weightGradient = input.Transpose().Dot(delta); // dE/dW
            Tensor biasGradient = delta.Sum(axis: 0);              // dE/db

            // 2. Обновление весов и смещений
            weights -= weightGradient * learningRate;             // W = W - η * dE/dW
            biases -= biasGradient * learningRate;                // b = b - η * dE/db

            // 3. Градиенты для предыдущего слоя
            Tensor previousDelta = delta.Dot(weights.Transpose()); // dE/dInput

            return previousDelta;
        }

        /// <summary>
        /// Тензор смещений для слоя.
        /// </summary>
        public Tensor biases;

        /// <summary>
        /// Тензор весов для слоя.
        /// </summary>
        public Tensor weights;

        /// <summary>
        /// Тензор ввода для обратного хода
        /// </summary>
        public Tensor input;

        /// <summary>
        /// Размер слоя (количество нейронов в слое).
        /// </summary>
        public int Size { get; set; }

        /// <summary>
        /// Размер входа (количество нейронов или признаков на входе).
        /// </summary>
        public int InputSize { get; set; }

        /// <summary>
        /// Случайный генератор для инициализации весов и смещений.
        /// </summary>
        private static Random random = new Random(DateTime.Now.Millisecond);
    }
}
