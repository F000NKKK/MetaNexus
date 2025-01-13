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
            this._weights = weights;
            this._biases = biases;
        }

        void ILayer.InitializeWeightsAndBiases()
        {
            _weights = new Tensor(new int[] { InputSize, Size });
            _biases = new Tensor(new int[] { Size });

            for (int i = 0; i < InputSize; i++)
            {
                for (int j = 0; j < Size; j++)
                {
                    _weights[i, j] = (float)(random.NextDouble() * 2.0 - 1.0);
                }
            }

            for (int i = 0; i < Size; i++)
            {
                _biases[i] = (float)(random.NextDouble() * 2.0 - 1.0);
            }
        }

        public Tensor GetWeights() => _weights;

        public void SetWeights(Tensor newWeights)
        {
            if (newWeights.Shape[0] != InputSize || newWeights.Shape[1] != Size)
                throw new ArgumentException("Размеры нового тензора весов не совпадают с размерами слоя.");

            _weights = newWeights;
        }

        public Tensor GetBiases() => _biases;

        public void SetBiases(Tensor newBiases)
        {
            if (newBiases.Shape[0] != Size)
                throw new ArgumentException("Размеры нового тензора смещений не совпадают с размерами слоя.");

            _biases = newBiases;
        }

        public abstract Tensor Forward(Tensor input);

        /// <summary>
        /// Тензор смещений для слоя.
        /// </summary>
        private Tensor _biases;

        /// <summary>
        /// Тензор весов для слоя.
        /// </summary>
        private Tensor _weights;

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
